#!/usr/bin/env python3
"""Deduplica y divide un dataset YOLO en train/val/test de forma reproducible.

El dataset original nunca se modifica. Las imagenes duplicadas se agrupan por SHA-256;
Los conflictos pueden resolverse por consenso geometrico y mediante correcciones externas
nombradas con el SHA-256 de la imagen. El dataset original nunca se modifica.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS_SALIDA = ("train", "val", "test")


@dataclass(frozen=True)
class Muestra:
    split_original: str
    imagen: Path
    etiqueta: Path
    hash_imagen: str
    firma_etiqueta: tuple[tuple[float, ...], ...]
    clases: tuple[int, ...]
    objetos: tuple[int, ...]


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crea una copia YOLO deduplicada con splits reproducibles."
    )
    parser.add_argument("--datos", type=Path, required=True, help="Dataset original.")
    parser.add_argument("--salida", type=Path, required=True, help="Nuevo dataset limpio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--splits-origen", nargs="+", default=["train", "valid", "val", "test"])
    parser.add_argument(
        "--resolver-conflictos",
        action="store_true",
        help="Resuelve diferencias geometricas compatibles mediante consenso.",
    )
    parser.add_argument(
        "--correcciones",
        type=Path,
        help="Directorio opcional con etiquetas corregidas llamadas <sha256>.txt.",
    )
    parser.add_argument(
        "--iou-minimo-conflicto",
        type=float,
        default=0.75,
        help="IoU minimo para resolver automaticamente cajas de igual clase (default: 0.75).",
    )
    return parser.parse_args()


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def leer_clases(ruta: Path) -> list[str]:
    clases = [x.strip() for x in ruta.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    if not clases:
        raise ValueError(f"Archivo de clases vacio: {ruta}")
    return clases


def leer_etiqueta(ruta: Path, numero_clases: int) -> tuple[tuple[tuple[float, ...], ...], tuple[int, ...]]:
    filas: list[tuple[float, ...]] = []
    objetos: list[int] = []
    for numero, linea in enumerate(ruta.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not linea.strip():
            continue
        partes = linea.split()
        if len(partes) != 5:
            raise ValueError(f"{ruta}, linea {numero}: formato YOLO invalido")
        valores = tuple(float(x) for x in partes)
        clase = int(valores[0])
        if valores[0] != clase or not 0 <= clase < numero_clases:
            raise ValueError(f"{ruta}, linea {numero}: clase invalida {valores[0]}")
        filas.append((float(clase), *(round(x, 6) for x in valores[1:])))
        objetos.append(clase)
    return tuple(sorted(filas)), tuple(objetos)


def inventariar(raiz: Path, splits: list[str], clases: list[str]) -> list[Muestra]:
    muestras: list[Muestra] = []
    vistos: set[Path] = set()
    for split in splits:
        dir_imagenes = raiz / split / "images"
        dir_etiquetas = raiz / split / "labels"
        if not dir_imagenes.is_dir() or not dir_etiquetas.is_dir():
            continue
        for imagen in sorted(dir_imagenes.iterdir()):
            if not imagen.is_file() or imagen.suffix.lower() not in EXTENSIONES:
                continue
            if imagen.resolve() in vistos:
                continue
            vistos.add(imagen.resolve())
            etiqueta = dir_etiquetas / f"{imagen.stem}.txt"
            if not etiqueta.exists():
                raise FileNotFoundError(f"Falta etiqueta para {imagen}")
            firma, objetos = leer_etiqueta(etiqueta, len(clases))
            muestras.append(
                Muestra(
                    split_original=split,
                    imagen=imagen,
                    etiqueta=etiqueta,
                    hash_imagen=sha256(imagen),
                    firma_etiqueta=firma,
                    clases=tuple(sorted(set(objetos))),
                    objetos=objetos,
                )
            )
    if not muestras:
        raise ValueError(f"No se encontraron imagenes en {raiz}")
    return muestras


def agrupar_duplicados(muestras: list[Muestra]) -> dict[str, list[Muestra]]:
    grupos: dict[str, list[Muestra]] = defaultdict(list)
    for muestra in muestras:
        grupos[muestra.hash_imagen].append(muestra)
    return dict(grupos)


def caja_xyxy(caja: tuple[float, ...]) -> tuple[float, float, float, float]:
    _, x, y, ancho, alto = caja
    return (x - ancho / 2, y - alto / 2, x + ancho / 2, y + alto / 2)


def iou_cajas(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = caja_xyxy(a)
    bx1, by1, bx2, by2 = caja_xyxy(b)
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ious_emparejados(a: Muestra, b: Muestra) -> list[float]:
    candidatos = []
    for ia, caja_a in enumerate(a.firma_etiqueta):
        for ib, caja_b in enumerate(b.firma_etiqueta):
            if int(caja_a[0]) == int(caja_b[0]):
                candidatos.append((iou_cajas(caja_a, caja_b), ia, ib))
    usados_a: set[int] = set()
    usados_b: set[int] = set()
    valores = []
    for valor, ia, ib in sorted(candidatos, reverse=True):
        if ia not in usados_a and ib not in usados_b:
            usados_a.add(ia)
            usados_b.add(ib)
            valores.append(valor)
    return valores


def muestra_con_etiqueta(base: Muestra, etiqueta: Path, numero_clases: int) -> Muestra:
    firma, objetos = leer_etiqueta(etiqueta, numero_clases)
    return Muestra(
        split_original=base.split_original,
        imagen=base.imagen,
        etiqueta=etiqueta,
        hash_imagen=base.hash_imagen,
        firma_etiqueta=firma,
        clases=tuple(sorted(set(objetos))),
        objetos=objetos,
    )


def resolver_grupos(
    grupos: dict[str, list[Muestra]],
    numero_clases: int,
    correcciones: Path | None,
    resolver_conflictos: bool,
    iou_minimo: float,
) -> tuple[list[Muestra], list[dict], list[dict]]:
    prioridad = {"train": 0, "valid": 1, "val": 1, "test": 2}
    elegidas: list[Muestra] = []
    resoluciones: list[dict] = []
    pendientes: list[dict] = []

    for hash_imagen, grupo in sorted(grupos.items()):
        ordenado = sorted(grupo, key=lambda m: (prioridad.get(m.split_original, 9), str(m.imagen)))
        firmas = {m.firma_etiqueta for m in grupo}
        correccion = correcciones / f"{hash_imagen}.txt" if correcciones else None

        if correccion and correccion.exists():
            elegida = muestra_con_etiqueta(ordenado[0], correccion, numero_clases)
            metodo = "correccion_manual"
            minimo_grupo = ""
        elif len(firmas) == 1:
            elegida = ordenado[0]
            metodo = "etiquetas_identicas" if len(grupo) > 1 else "imagen_unica"
            minimo_grupo = "1.000000"
        elif resolver_conflictos:
            conteos = [Counter(m.objetos) for m in grupo]
            compatibles = all(c == conteos[0] for c in conteos[1:])
            pares: dict[tuple[int, int], list[float]] = {}
            for ia in range(len(grupo)):
                for ib in range(ia + 1, len(grupo)):
                    pares[(ia, ib)] = ious_emparejados(grupo[ia], grupo[ib])
            valores = [v for lista in pares.values() for v in lista]
            minimo = min(valores) if valores else 0.0
            if compatibles and minimo >= iou_minimo:
                puntuaciones = []
                for indice, muestra in enumerate(grupo):
                    propios = []
                    for (ia, ib), lista in pares.items():
                        if indice in (ia, ib):
                            propios.extend(lista)
                    promedio = sum(propios) / len(propios) if propios else 0.0
                    puntuaciones.append((promedio, -prioridad.get(muestra.split_original, 9), str(muestra.imagen), muestra))
                elegida = max(puntuaciones, key=lambda x: (x[0], x[1], x[2]))[3]
                metodo = "consenso_geometrico"
                minimo_grupo = f"{minimo:.6f}"
            else:
                pendientes.append({
                    "sha256": hash_imagen,
                    "motivo": "clases_o_cantidades_diferentes" if not compatibles else "iou_inferior_al_umbral",
                    "iou_minimo": f"{minimo:.6f}",
                    "archivos": " | ".join(str(m.imagen) for m in grupo),
                    "etiquetas": " | ".join(str(m.etiqueta) for m in grupo),
                })
                continue
        else:
            pendientes.append({
                "sha256": hash_imagen,
                "motivo": "resolucion_no_habilitada",
                "iou_minimo": "",
                "archivos": " | ".join(str(m.imagen) for m in grupo),
                "etiquetas": " | ".join(str(m.etiqueta) for m in grupo),
            })
            continue

        elegidas.append(elegida)
        if len(grupo) > 1:
            resoluciones.append({
                "sha256": hash_imagen,
                "copias": len(grupo),
                "metodo": metodo,
                "iou_minimo": minimo_grupo,
                "imagen_elegida": str(elegida.imagen),
                "etiqueta_elegida": str(elegida.etiqueta),
                "objetos": len(elegida.objetos),
            })
    return elegidas, resoluciones, pendientes


def escribir_csv(ruta: Path, filas: list[dict], columnas: list[str]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(filas)


def revisar_conflictos(grupos: dict[str, list[Muestra]]) -> list[dict[str, str]]:
    conflictos = []
    for hash_imagen, grupo in grupos.items():
        firmas = {m.firma_etiqueta for m in grupo}
        if len(firmas) <= 1:
            continue
        conflictos.append(
            {
                "sha256": hash_imagen,
                "archivos": " | ".join(str(m.imagen) for m in grupo),
                "etiquetas": " | ".join(str(m.etiqueta) for m in grupo),
                "variantes": str(len(firmas)),
            }
        )
    return conflictos


def objetivos_splits(total: int, proporciones: dict[str, float]) -> dict[str, int]:
    crudos = {s: total * proporciones[s] for s in SPLITS_SALIDA}
    objetivos = {s: int(crudos[s]) for s in SPLITS_SALIDA}
    faltan = total - sum(objetivos.values())
    orden = sorted(SPLITS_SALIDA, key=lambda s: (crudos[s] - objetivos[s], -SPLITS_SALIDA.index(s)), reverse=True)
    for split in orden[:faltan]:
        objetivos[split] += 1
    return objetivos


def dividir(
    muestras: list[Muestra],
    numero_clases: int,
    proporciones: dict[str, float],
    seed: int,
) -> dict[str, list[Muestra]]:
    rng = random.Random(seed)
    total_objetos = Counter(c for m in muestras for c in m.objetos)
    frecuencia_imagen = Counter(c for m in muestras for c in m.clases)
    objetivo_imagenes = objetivos_splits(len(muestras), proporciones)
    objetivo_clases = {
        split: {c: total_objetos[c] * proporciones[split] for c in range(numero_clases)}
        for split in SPLITS_SALIDA
    }

    ordenadas = muestras[:]
    rng.shuffle(ordenadas)
    ordenadas.sort(
        key=lambda m: (
            min((frecuencia_imagen[c] for c in m.clases), default=10**9),
            -len(m.objetos),
        )
    )

    asignadas: dict[str, list[Muestra]] = {s: [] for s in SPLITS_SALIDA}
    conteo_clases = {s: Counter() for s in SPLITS_SALIDA}

    for muestra in ordenadas:
        candidatos = [s for s in SPLITS_SALIDA if len(asignadas[s]) < objetivo_imagenes[s]]
        if not candidatos:
            raise RuntimeError("No quedan splits con capacidad")
        puntuaciones = []
        objetos_muestra = Counter(muestra.objetos)
        for split in candidatos:
            deficit_imagen = (objetivo_imagenes[split] - len(asignadas[split])) / max(objetivo_imagenes[split], 1)
            deficit_clases = 0.0
            for clase, cantidad in objetos_muestra.items():
                objetivo = max(objetivo_clases[split][clase], 1.0)
                deficit_clases += max(0.0, objetivo - conteo_clases[split][clase]) / objetivo * cantidad
            puntuaciones.append((deficit_clases + 0.20 * deficit_imagen, rng.random(), split))
        split_elegido = max(puntuaciones)[2]
        asignadas[split_elegido].append(muestra)
        conteo_clases[split_elegido].update(muestra.objetos)

    return asignadas


def nombre_destino(muestra: Muestra) -> str:
    nombre = "".join(c if c.isalnum() or c in "._-" else "_" for c in muestra.imagen.name)
    return f"{muestra.hash_imagen[:12]}__{nombre}"


def main() -> int:
    args = argumentos()
    raiz = args.datos.resolve()
    salida = args.salida.resolve()
    proporciones = {"train": args.train, "val": args.val, "test": args.test}

    if abs(sum(proporciones.values()) - 1.0) > 1e-9 or any(v <= 0 for v in proporciones.values()):
        raise ValueError("Las proporciones train/val/test deben ser positivas y sumar 1")
    if salida.exists():
        raise FileExistsError(f"La salida ya existe; no se sobrescribira: {salida}")
    clases = leer_clases(raiz / "classes.txt")
    muestras = inventariar(raiz, args.splits_origen, clases)
    grupos = agrupar_duplicados(muestras)
    if not 0 <= args.iou_minimo_conflicto <= 1:
        raise ValueError("--iou-minimo-conflicto debe estar entre 0 y 1")
    correcciones = args.correcciones.resolve() if args.correcciones else None
    if correcciones and not correcciones.is_dir():
        raise FileNotFoundError(f"No existe el directorio de correcciones: {correcciones}")
    unicas, resoluciones, pendientes = resolver_grupos(
        grupos,
        len(clases),
        correcciones,
        args.resolver_conflictos,
        args.iou_minimo_conflicto,
    )

    if pendientes:
        salida.mkdir(parents=True)
        escribir_csv(
            salida / "conflictos_duplicados.csv",
            pendientes,
            ["sha256", "motivo", "iou_minimo", "archivos", "etiquetas"],
        )
        print(f"ERROR: quedan {len(pendientes)} grupos duplicados sin resolver.")
        print(f"Revisar: {salida / 'conflictos_duplicados.csv'}")
        return 2
    asignadas = dividir(unicas, len(clases), proporciones, args.seed)

    for split in SPLITS_SALIDA:
        (salida / "images" / split).mkdir(parents=True)
        (salida / "labels" / split).mkdir(parents=True)

    manifiesto = []
    for split, elementos in asignadas.items():
        for muestra in sorted(elementos, key=lambda m: m.hash_imagen):
            nombre = nombre_destino(muestra)
            destino_imagen = salida / "images" / split / nombre
            destino_etiqueta = salida / "labels" / split / f"{Path(nombre).stem}.txt"
            shutil.copy2(muestra.imagen, destino_imagen)
            shutil.copy2(muestra.etiqueta, destino_etiqueta)
            manifiesto.append(
                {
                    "sha256": muestra.hash_imagen,
                    "split_destino": split,
                    "imagen_destino": str(destino_imagen.relative_to(salida)),
                    "split_original": muestra.split_original,
                    "imagen_original": str(muestra.imagen),
                    "clases": ",".join(str(c) for c in muestra.clases),
                    "objetos": len(muestra.objetos),
                    "copias_originales": len(grupos[muestra.hash_imagen]),
                }
            )

    duplicados = []
    for hash_imagen, grupo in sorted(grupos.items()):
        if len(grupo) > 1:
            duplicados.append(
                {
                    "sha256": hash_imagen,
                    "cantidad": len(grupo),
                    "splits_originales": ",".join(sorted({m.split_original for m in grupo})),
                    "archivos": " | ".join(str(m.imagen) for m in grupo),
                }
            )

    escribir_csv(
        salida / "manifiesto_splits.csv",
        manifiesto,
        ["sha256", "split_destino", "imagen_destino", "split_original", "imagen_original", "clases", "objetos", "copias_originales"],
    )
    escribir_csv(
        salida / "duplicados_eliminados.csv",
        duplicados,
        ["sha256", "cantidad", "splits_originales", "archivos"],
    )
    escribir_csv(
        salida / "resoluciones_duplicados.csv",
        resoluciones,
        ["sha256", "copias", "metodo", "iou_minimo", "imagen_elegida", "etiqueta_elegida", "objetos"],
    )

    (salida / "classes.txt").write_text("\n".join(clases) + "\n", encoding="utf-8")
    yaml = ["path: .", "train: images/train", "val: images/val", "test: images/test", "", "names:"]
    yaml.extend(f"  {i}: {nombre}" for i, nombre in enumerate(clases))
    (salida / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    resumen = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_original": str(raiz),
        "dataset_salida": str(salida),
        "seed": args.seed,
        "proporciones": proporciones,
        "imagenes_originales": len(muestras),
        "imagenes_unicas": len(unicas),
        "copias_eliminadas": len(muestras) - len(unicas),
        "grupos_duplicados": len(duplicados),
        "resoluciones_duplicados": dict(Counter(r["metodo"] for r in resoluciones)),
        "umbral_iou_conflictos": args.iou_minimo_conflicto,
        "splits": {},
    }
    for split, elementos in asignadas.items():
        conteo = Counter(c for m in elementos for c in m.objetos)
        resumen["splits"][split] = {
            "imagenes": len(elementos),
            "objetos": sum(conteo.values()),
            "objetos_por_clase": {clases[i]: conteo[i] for i in range(len(clases))},
        }
    (salida / "resumen_preparacion.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Imagenes originales: {len(muestras)}")
    print(f"Imagenes unicas: {len(unicas)}")
    print(f"Copias eliminadas: {len(muestras) - len(unicas)}")
    for split in SPLITS_SALIDA:
        print(f"{split}: {len(asignadas[split])} imagenes")
    print(f"Dataset limpio: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
