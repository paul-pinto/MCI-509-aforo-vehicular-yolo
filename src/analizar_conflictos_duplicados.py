#!/usr/bin/env python3
"""Diagnostica etiquetas incompatibles asociadas a imágenes duplicadas.

No modifica el dataset. Lee el CSV generado por preparar_dataset.py y compara
las variantes de etiquetas YOLO mediante clases, cantidad de objetos e IoU.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conflictos", required=True, type=Path)
    parser.add_argument("--clases", required=True, type=Path)
    parser.add_argument("--salida", required=True, type=Path)
    return parser.parse_args()


def leer_clases(ruta: Path) -> list[str]:
    return [linea.strip() for linea in ruta.read_text(encoding="utf-8-sig").splitlines() if linea.strip()]


def leer_etiqueta(ruta: Path) -> list[dict]:
    cajas = []
    for numero, linea in enumerate(ruta.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not linea.strip():
            continue
        partes = linea.split()
        if len(partes) != 5:
            raise ValueError(f"{ruta}, línea {numero}: se esperaban 5 valores")
        clase = int(partes[0])
        x, y, ancho, alto = map(float, partes[1:])
        cajas.append({"clase": clase, "x": x, "y": y, "w": ancho, "h": alto})
    return cajas


def xyxy(caja: dict) -> tuple[float, float, float, float]:
    return (
        caja["x"] - caja["w"] / 2,
        caja["y"] - caja["h"] / 2,
        caja["x"] + caja["w"] / 2,
        caja["y"] + caja["h"] / 2,
    )


def iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def emparejar_por_iou(a: list[dict], b: list[dict]) -> list[float]:
    """Emparejamiento voraz por mayor IoU, restringido a la misma clase."""
    candidatos = []
    for ia, caja_a in enumerate(a):
        for ib, caja_b in enumerate(b):
            if caja_a["clase"] == caja_b["clase"]:
                candidatos.append((iou(caja_a, caja_b), ia, ib))
    usados_a, usados_b, valores = set(), set(), []
    for valor, ia, ib in sorted(candidatos, reverse=True):
        if ia not in usados_a and ib not in usados_b:
            usados_a.add(ia)
            usados_b.add(ib)
            valores.append(valor)
    return valores


def nombres_clases(cajas: list[dict], clases: list[str]) -> str:
    contador = Counter(c["clase"] for c in cajas)
    partes = []
    for indice, cantidad in sorted(contador.items()):
        nombre = clases[indice] if 0 <= indice < len(clases) else f"clase_{indice}"
        partes.append(f"{nombre}:{cantidad}")
    return "; ".join(partes) or "sin objetos"


def main() -> None:
    args = argumentos()
    conflictos = args.conflictos.resolve()
    clases = leer_clases(args.clases.resolve())
    salida = args.salida.resolve()
    salida.mkdir(parents=True, exist_ok=True)

    diagnosticos, variantes_salida = [], []
    with conflictos.open(encoding="utf-8-sig", newline="") as archivo:
        filas = list(csv.DictReader(archivo))

    for indice, fila in enumerate(filas, 1):
        rutas = [Path(p.strip()) for p in fila["etiquetas"].split(" | ") if p.strip()]
        variantes = []
        for numero_variante, ruta in enumerate(rutas, 1):
            if not ruta.exists():
                raise FileNotFoundError(f"No se encontró la etiqueta indicada en el CSV: {ruta}")
            cajas = leer_etiqueta(ruta)
            variantes.append(cajas)
            variantes_salida.append({
                "grupo": indice,
                "sha256": fila["sha256"],
                "variante": numero_variante,
                "ruta_etiqueta": str(ruta),
                "cantidad_objetos": len(cajas),
                "clases": nombres_clases(cajas, clases),
                "cajas_yolo_json": json.dumps(cajas, ensure_ascii=False),
            })

        multisets = [Counter(c["clase"] for c in cajas) for cajas in variantes]
        mismas_clases = all(m == multisets[0] for m in multisets[1:])
        misma_cantidad = len({len(cajas) for cajas in variantes}) == 1
        valores_iou = []
        for a, b in itertools.combinations(variantes, 2):
            valores_iou.extend(emparejar_por_iou(a, b))
        minimo_iou = min(valores_iou) if valores_iou else None
        promedio_iou = mean(valores_iou) if valores_iou else None

        if not misma_cantidad:
            categoria = "cantidad_objetos_diferente"
        elif not mismas_clases:
            categoria = "clases_diferentes"
        elif minimo_iou is not None and minimo_iou >= 0.95:
            categoria = "cajas_casi_iguales"
        elif minimo_iou is not None and minimo_iou >= 0.80:
            categoria = "cajas_similares"
        else:
            categoria = "cajas_diferentes"

        diagnosticos.append({
            "grupo": indice,
            "sha256": fila["sha256"],
            "archivos": fila["archivos"],
            "cantidad_variantes": len(variantes),
            "objetos_por_variante": " | ".join(str(len(v)) for v in variantes),
            "clases_por_variante": " | ".join(nombres_clases(v, clases) for v in variantes),
            "misma_cantidad": misma_cantidad,
            "mismas_clases": mismas_clases,
            "iou_minimo": "" if minimo_iou is None else f"{minimo_iou:.6f}",
            "iou_promedio": "" if promedio_iou is None else f"{promedio_iou:.6f}",
            "categoria": categoria,
        })

    campos_diag = list(diagnosticos[0].keys()) if diagnosticos else ["grupo", "sha256", "categoria"]
    with (salida / "diagnostico_conflictos.csv").open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos_diag)
        escritor.writeheader()
        escritor.writerows(diagnosticos)

    campos_var = list(variantes_salida[0].keys()) if variantes_salida else ["grupo", "sha256", "variante"]
    with (salida / "variantes_conflictos.csv").open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos_var)
        escritor.writeheader()
        escritor.writerows(variantes_salida)

    conteo = Counter(d["categoria"] for d in diagnosticos)
    resumen = {
        "archivo_conflictos": str(conflictos),
        "grupos_analizados": len(diagnosticos),
        "categorias": dict(sorted(conteo.items())),
        "criterios_iou": {"casi_iguales": 0.95, "similares": 0.80},
        "nota": "Diagnóstico solamente; ningún archivo del dataset fue modificado.",
    }
    (salida / "resumen_conflictos.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Grupos analizados: {len(diagnosticos)}")
    for categoria, cantidad in sorted(conteo.items()):
        print(f"{categoria}: {cantidad}")
    print(f"Resultados: {salida}")


if __name__ == "__main__":
    main()
