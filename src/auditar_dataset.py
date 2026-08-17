#!/usr/bin/env python3
"""Audita un dataset de deteccion en formato YOLO sin modificarlo.

Genera tres evidencias reproducibles:
  - auditoria_dataset.json: resumen completo y parametros de ejecucion.
  - auditoria_clases.csv: distribucion de objetos por clase y split.
  - auditoria_problemas.csv: archivos y anotaciones que requieren revision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError


EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TOLERANCIA_BORDE = 1e-6


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auditoria reproducible de imagenes y etiquetas YOLO."
    )
    parser.add_argument(
        "--datos",
        type=Path,
        required=True,
        help="Raiz YOLO con <split>/images o images/<split>.",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=Path("resultados/auditoria_dataset"),
        help="Directorio donde se guardaran JSON y CSV.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "valid", "test"],
        help="Splits que se intentaran auditar.",
    )
    parser.add_argument(
        "--clases",
        type=Path,
        help="Archivo de clases. Por defecto usa <datos>/classes.txt.",
    )
    return parser.parse_args()


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def cargar_clases(ruta: Path) -> list[str]:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo de clases: {ruta}")
    clases = [linea.strip() for linea in ruta.read_text(encoding="utf-8-sig").splitlines()]
    clases = [clase for clase in clases if clase]
    if not clases:
        raise ValueError(f"El archivo de clases esta vacio: {ruta}")
    return clases


def agregar_problema(
    problemas: list[dict[str, str]],
    split: str,
    archivo: Path | str,
    tipo: str,
    detalle: str,
) -> None:
    problemas.append(
        {
            "split": split,
            "archivo": str(archivo),
            "tipo": tipo,
            "detalle": detalle,
        }
    )


def leer_etiqueta(
    ruta: Path,
    split: str,
    nombres_clases: list[str],
    problemas: list[dict[str, str]],
) -> tuple[Counter[int], int, int]:
    conteo: Counter[int] = Counter()
    cajas_validas = 0
    cajas_invalidas = 0

    try:
        lineas = ruta.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as error:
        agregar_problema(problemas, split, ruta, "etiqueta_no_legible", str(error))
        return conteo, 0, 1

    if not any(linea.strip() for linea in lineas):
        agregar_problema(problemas, split, ruta, "etiqueta_vacia", "Sin objetos anotados")
        return conteo, 0, 0

    for numero_linea, contenido in enumerate(lineas, start=1):
        contenido = contenido.strip()
        if not contenido:
            continue
        partes = contenido.split()
        if len(partes) != 5:
            cajas_invalidas += 1
            agregar_problema(
                problemas,
                split,
                ruta,
                "formato_yolo_invalido",
                f"Linea {numero_linea}: se esperaban 5 valores y hay {len(partes)}",
            )
            continue
        try:
            clase_float = float(partes[0])
            clase_id = int(clase_float)
            coordenadas = [float(valor) for valor in partes[1:]]
        except ValueError:
            cajas_invalidas += 1
            agregar_problema(
                problemas, split, ruta, "valor_no_numerico", f"Linea {numero_linea}"
            )
            continue

        if clase_float != clase_id or not 0 <= clase_id < len(nombres_clases):
            cajas_invalidas += 1
            agregar_problema(
                problemas,
                split,
                ruta,
                "clase_invalida",
                f"Linea {numero_linea}: class_id={partes[0]}",
            )
            continue

        x, y, ancho, alto = coordenadas
        if not all(0.0 <= valor <= 1.0 for valor in coordenadas):
            cajas_invalidas += 1
            agregar_problema(
                problemas,
                split,
                ruta,
                "coordenada_fuera_de_rango",
                f"Linea {numero_linea}: {coordenadas}",
            )
            continue
        if ancho <= 0 or alto <= 0:
            cajas_invalidas += 1
            agregar_problema(
                problemas,
                split,
                ruta,
                "caja_sin_area",
                f"Linea {numero_linea}: ancho={ancho}, alto={alto}",
            )
            continue
        if (
            x - ancho / 2 < -TOLERANCIA_BORDE
            or x + ancho / 2 > 1 + TOLERANCIA_BORDE
            or y - alto / 2 < -TOLERANCIA_BORDE
            or y + alto / 2 > 1 + TOLERANCIA_BORDE
        ):
            cajas_invalidas += 1
            agregar_problema(
                problemas,
                split,
                ruta,
                "caja_fuera_de_imagen",
                f"Linea {numero_linea}: x={x}, y={y}, ancho={ancho}, alto={alto}",
            )
            continue

        conteo[clase_id] += 1
        cajas_validas += 1

    return conteo, cajas_validas, cajas_invalidas


def auditar_split(
    raiz: Path,
    split: str,
    clases: list[str],
    problemas: list[dict[str, str]],
    hashes: dict[str, list[dict[str, str]]],
) -> dict:
    # Admite las dos organizaciones habituales de datasets YOLO:
    #   train/images + train/labels
    #   images/train + labels/train (estructura generada por preparar_dataset.py)
    candidatos = [
        (raiz / split / "images", raiz / split / "labels"),
        (raiz / "images" / split, raiz / "labels" / split),
    ]
    dir_imagenes, dir_etiquetas = next(
        ((imagenes, etiquetas) for imagenes, etiquetas in candidatos if imagenes.is_dir() and etiquetas.is_dir()),
        candidatos[0],
    )

    if not dir_imagenes.is_dir() or not dir_etiquetas.is_dir():
        return {"presente": False}

    imagenes = sorted(
        ruta for ruta in dir_imagenes.iterdir() if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN
    )
    etiquetas = sorted(ruta for ruta in dir_etiquetas.glob("*.txt") if ruta.is_file())
    etiquetas_por_stem = {ruta.stem: ruta for ruta in etiquetas}
    imagenes_por_stem = {ruta.stem: ruta for ruta in imagenes}

    conteo_clases: Counter[int] = Counter()
    resoluciones: Counter[str] = Counter()
    cajas_validas = 0
    cajas_invalidas = 0
    imagenes_no_legibles = 0

    for imagen in imagenes:
        try:
            with Image.open(imagen) as archivo_imagen:
                archivo_imagen.verify()
            with Image.open(imagen) as archivo_imagen:
                ancho, alto = archivo_imagen.size
        except (UnidentifiedImageError, OSError, ValueError) as error:
            imagenes_no_legibles += 1
            agregar_problema(problemas, split, imagen, "imagen_no_legible", str(error))
        else:
            resoluciones[f"{ancho}x{alto}"] += 1

        hashes[sha256(imagen)].append({"split": split, "archivo": str(imagen)})

        etiqueta = etiquetas_por_stem.get(imagen.stem)
        if etiqueta is None:
            agregar_problema(problemas, split, imagen, "etiqueta_faltante", "No existe .txt correspondiente")
            continue
        conteo, validas, invalidas = leer_etiqueta(etiqueta, split, clases, problemas)
        conteo_clases.update(conteo)
        cajas_validas += validas
        cajas_invalidas += invalidas

    for stem, etiqueta in etiquetas_por_stem.items():
        if stem not in imagenes_por_stem:
            agregar_problema(problemas, split, etiqueta, "imagen_faltante", "No existe imagen correspondiente")

    return {
        "presente": True,
        "imagenes": len(imagenes),
        "etiquetas": len(etiquetas),
        "imagenes_no_legibles": imagenes_no_legibles,
        "cajas_validas": cajas_validas,
        "cajas_invalidas": cajas_invalidas,
        "objetos_por_clase": {clases[i]: conteo_clases[i] for i in range(len(clases))},
        "resoluciones": dict(resoluciones.most_common()),
    }


def escribir_csv(ruta: Path, filas: list[dict], columnas: list[str]) -> None:
    with ruta.open("w", newline="", encoding="utf-8-sig") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(filas)


def main() -> int:
    args = argumentos()
    raiz = args.datos.resolve()
    salida = args.salida.resolve()
    archivo_clases = (args.clases or raiz / "classes.txt").resolve()

    if not raiz.is_dir():
        raise NotADirectoryError(f"No existe el dataset: {raiz}")

    clases = cargar_clases(archivo_clases)
    problemas: list[dict[str, str]] = []
    hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    resumen_splits = {
        split: auditar_split(raiz, split, clases, problemas, hashes)
        for split in args.splits
    }

    duplicados = [archivos for archivos in hashes.values() if len(archivos) > 1]
    for grupo, archivos in enumerate(duplicados, start=1):
        detalle = "; ".join(f"{item['split']}:{item['archivo']}" for item in archivos)
        for item in archivos:
            agregar_problema(
                problemas,
                item["split"],
                item["archivo"],
                "imagen_duplicada",
                f"Grupo {grupo}: {detalle}",
            )

    salida.mkdir(parents=True, exist_ok=True)
    auditoria = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(raiz),
        "archivo_clases": str(archivo_clases),
        "clases": clases,
        "splits_solicitados": args.splits,
        "splits": resumen_splits,
        "grupos_imagenes_duplicadas": len(duplicados),
        "problemas_totales": len(problemas),
        "entorno": {
            "python": sys.version.split()[0],
            "pillow": Image.__version__,
            "sistema": platform.platform(),
        },
    }

    ruta_json = salida / "auditoria_dataset.json"
    ruta_json.write_text(json.dumps(auditoria, indent=2, ensure_ascii=False), encoding="utf-8")

    filas_clases = []
    for split, datos in resumen_splits.items():
        if not datos.get("presente"):
            continue
        total = sum(datos["objetos_por_clase"].values())
        for clase_id, clase in enumerate(clases):
            cantidad = datos["objetos_por_clase"][clase]
            filas_clases.append(
                {
                    "split": split,
                    "clase_id": clase_id,
                    "clase": clase,
                    "objetos": cantidad,
                    "porcentaje_split": round(100 * cantidad / total, 4) if total else 0,
                }
            )
    escribir_csv(
        salida / "auditoria_clases.csv",
        filas_clases,
        ["split", "clase_id", "clase", "objetos", "porcentaje_split"],
    )
    escribir_csv(
        salida / "auditoria_problemas.csv",
        problemas,
        ["split", "archivo", "tipo", "detalle"],
    )

    print(f"Dataset: {raiz}")
    print(f"Clases: {len(clases)} -> {', '.join(clases)}")
    for split, datos in resumen_splits.items():
        if datos.get("presente"):
            print(
                f"{split}: {datos['imagenes']} imagenes, {datos['etiquetas']} etiquetas, "
                f"{datos['cajas_validas']} cajas validas, {datos['cajas_invalidas']} invalidas"
            )
        else:
            print(f"{split}: no presente")
    print(f"Duplicados exactos: {len(duplicados)} grupos")
    print(f"Problemas registrados: {len(problemas)}")
    print(f"Resultados: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
