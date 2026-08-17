#!/usr/bin/env python3
"""Evalua una sola vez un detector YOLO sobre el split test y guarda evidencias."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Pesos best.pt.")
    parser.add_argument("--data", type=Path, required=True, help="data.yaml del dataset.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("resultados/evaluacion"))
    parser.add_argument("--name", default="test_yolo11n_definitivo")
    return parser.parse_args()


def sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def yaml_resuelto(data_yaml: Path, destino: Path) -> tuple[Path, dict]:
    import yaml

    contenido = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    if not isinstance(contenido, dict):
        raise ValueError("data.yaml invalido")
    faltantes = [x for x in ("train", "val", "test", "names") if x not in contenido]
    if faltantes:
        raise ValueError(f"Faltan campos en data.yaml: {', '.join(faltantes)}")
    base = Path(str(contenido.get("path", ".")))
    base = base if base.is_absolute() else data_yaml.parent / base
    base = base.resolve()
    split_test = Path(str(contenido["test"]))
    split_test = split_test if split_test.is_absolute() else base / split_test
    if not split_test.resolve().is_dir():
        raise FileNotFoundError(f"No existe el split test: {split_test.resolve()}")
    contenido["path"] = base.as_posix()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        yaml.safe_dump(contenido, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return destino, contenido


def numero(valor) -> float:
    return float(valor.item() if hasattr(valor, "item") else valor)


def main() -> int:
    args = argumentos()
    modelo_path = args.model.resolve()
    data_path = args.data.resolve()
    if not modelo_path.is_file():
        raise FileNotFoundError(f"No existen los pesos: {modelo_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"No existe data.yaml: {data_path}")
    if args.imgsz <= 0 or args.batch == 0 or args.workers < 0:
        raise ValueError("imgsz debe ser positivo, batch no puede ser 0 y workers no puede ser negativo")

    proyecto = args.project.resolve()
    data_efectivo, contenido = yaml_resuelto(
        data_path, proyecto / "_config" / "data_test_resuelto.yaml"
    )

    import torch
    import ultralytics
    from ultralytics import YOLO

    print("ATENCION: se evaluara el conjunto TEST reservado.")
    print(f"Modelo: {modelo_path}")
    print(f"SHA-256: {sha256(modelo_path)}")
    print(f"Dataset: {data_efectivo}")

    modelo = YOLO(str(modelo_path))
    metricas = modelo.val(
        data=str(data_efectivo),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(proyecto),
        name=args.name,
        exist_ok=False,
        plots=True,
        save_json=False,
        verbose=True,
    )

    carpeta = Path(metricas.save_dir).resolve()
    nombres = modelo.names
    filas = []
    for indice in range(len(nombres)):
        precision = numero(metricas.box.p[indice])
        recall = numero(metricas.box.r[indice])
        ap50 = numero(metricas.box.ap50[indice])
        map_50_95 = numero(metricas.box.maps[indice])
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        filas.append(
            {
                "clase_id": indice,
                "clase": nombres[indice],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "map50": ap50,
                "map50_95": map_50_95,
            }
        )

    with (carpeta / "metricas_test_por_clase.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)

    resumen = {
        "evaluado_utc": datetime.now(timezone.utc).isoformat(),
        "advertencia_metodologica": "El split test se utiliza solo para evaluacion final; no ajustar hiperparametros con estos resultados.",
        "split": args.split,
        "modelo": str(modelo_path),
        "modelo_sha256": sha256(modelo_path),
        "data_yaml_original": str(data_path),
        "data_yaml_efectivo": str(data_efectivo),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "metricas_globales": {
            "precision": numero(metricas.box.mp),
            "recall": numero(metricas.box.mr),
            "map50": numero(metricas.box.map50),
            "map50_95": numero(metricas.box.map),
        },
        "velocidad_ms_por_imagen": {k: numero(v) for k, v in metricas.speed.items()},
        "metricas_por_clase": filas,
        "entorno": {
            "python": sys.version.split()[0],
            "sistema": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_disponible": torch.cuda.is_available(),
            "cuda_runtime_torch": torch.version.cuda,
        },
        "clases_configuradas": contenido["names"],
        "directorio_resultados": str(carpeta),
    }
    (carpeta / "metricas_test.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Resultados: {carpeta}")
    print(f"mAP50: {resumen['metricas_globales']['map50']:.6f}")
    print(f"mAP50-95: {resumen['metricas_globales']['map50_95']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
