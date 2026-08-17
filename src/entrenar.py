#!/usr/bin/env python3
"""Fine-tuning reproducible de YOLO11 para deteccion vehicular."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Ruta a data.yaml.")
    parser.add_argument("--model", default="yolo11n.pt", help="Pesos iniciales o YAML del modelo.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto, cpu, 0, 0,1, etc.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=Path("runs/entrenamiento"))
    parser.add_argument("--name", default="yolo11n_vehiculos_seed42")
    parser.add_argument("--freeze", type=int, default=0, help="Capas iniciales que se congelaran.")
    parser.add_argument("--prueba", action="store_true", help="Ejecuta 1 epoca con 5%% del train.")
    return parser.parse_args()


def validar_argumentos(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch == 0 or args.workers < 0:
        raise ValueError("epochs/imgsz deben ser positivos, batch no puede ser 0 y workers no puede ser negativo")
    if args.patience < 0 or args.freeze < 0:
        raise ValueError("patience y freeze no pueden ser negativos")


def validar_dataset(data_yaml: Path) -> tuple[dict, Path]:
    import yaml

    if not data_yaml.is_file():
        raise FileNotFoundError(f"No existe data.yaml: {data_yaml}")
    contenido = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    if not isinstance(contenido, dict):
        raise ValueError("data.yaml no contiene un diccionario valido")
    faltantes = [clave for clave in ("train", "val", "test", "names") if clave not in contenido]
    if faltantes:
        raise ValueError(f"Faltan campos en data.yaml: {', '.join(faltantes)}")

    base_declarada = Path(str(contenido.get("path", ".")))
    base = base_declarada if base_declarada.is_absolute() else data_yaml.parent / base_declarada
    for split in ("train", "val", "test"):
        ruta = Path(str(contenido[split]))
        ruta = ruta if ruta.is_absolute() else base / ruta
        if not ruta.resolve().is_dir():
            raise FileNotFoundError(f"No existe el directorio {split}: {ruta.resolve()}")
    nombres = contenido["names"]
    if not isinstance(nombres, (dict, list)) or len(nombres) == 0:
        raise ValueError("El campo names de data.yaml esta vacio o es invalido")
    return contenido, base.resolve()


def crear_yaml_resuelto(contenido: dict, base: Path, destino: Path) -> Path:
    import yaml

    resuelto = dict(contenido)
    resuelto["path"] = base.as_posix()
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        yaml.safe_dump(resuelto, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return destino


def resolver_dispositivo(solicitado: str, torch) -> str:
    if solicitado.lower() != "auto":
        return solicitado
    return "0" if torch.cuda.is_available() else "cpu"


def informacion_gpu(torch, dispositivo: str) -> dict | None:
    if dispositivo == "cpu" or not torch.cuda.is_available():
        return None
    indice = int(dispositivo.split(",")[0])
    propiedades = torch.cuda.get_device_properties(indice)
    return {
        "indice": indice,
        "nombre": propiedades.name,
        "memoria_gib": round(propiedades.total_memory / 1024**3, 3),
        "capacidad_cuda": f"{propiedades.major}.{propiedades.minor}",
    }


def guardar_json(ruta: Path, contenido: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = argumentos()
    validar_argumentos(args)
    data_yaml = args.data.resolve()
    dataset, base_dataset = validar_dataset(data_yaml)

    # expandable_segments no esta soportado por el asignador CUDA de Windows/WDDM.
    # Solo se solicita en plataformas donde PyTorch puede aprovecharlo.
    if os.name != "nt":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    dispositivo = resolver_dispositivo(args.device, torch)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    epochs = 1 if args.prueba else args.epochs
    fraction = 0.05 if args.prueba else 1.0
    nombre = f"{args.name}_prueba" if args.prueba else args.name
    proyecto = args.project.resolve()
    data_yaml_resuelto = crear_yaml_resuelto(
        dataset,
        base_dataset,
        proyecto / "_config" / f"data_resuelto_{args.seed}.yaml",
    )
    gpu = informacion_gpu(torch, dispositivo)

    configuracion = {
        "inicio_utc": datetime.now(timezone.utc).isoformat(),
        "tipo_entrenamiento": "fine_tuning_transfer_learning",
        "pesos_iniciales": args.model,
        "data_yaml": str(data_yaml),
        "data_yaml_resuelto": str(data_yaml_resuelto),
        "raiz_dataset_resuelta": str(base_dataset),
        "clases": dataset["names"],
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": dispositivo,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "freeze": args.freeze,
        "fraction": fraction,
        "deterministic": True,
        "amp": dispositivo != "cpu",
        "entorno": {
            "python": sys.version.split()[0],
            "sistema": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_disponible": torch.cuda.is_available(),
            "cuda_runtime_torch": torch.version.cuda,
            "gpu": gpu,
        },
    }

    print(f"Dataset: {data_yaml}")
    print(f"Modelo inicial: {args.model}")
    print(f"Dispositivo: {dispositivo}")
    if gpu:
        print(f"GPU: {gpu['nombre']} | VRAM: {gpu['memoria_gib']} GiB")
    print(f"Configuracion: epochs={epochs}, imgsz={args.imgsz}, batch={args.batch}, workers={args.workers}")
    if args.prueba:
        print("Modo prueba: 1 epoca y 5% del conjunto de entrenamiento")

    modelo = YOLO(args.model)
    try:
        resultados = modelo.train(
            data=str(data_yaml_resuelto),
            epochs=epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=dispositivo,
            workers=args.workers,
            project=str(proyecto),
            name=nombre,
            exist_ok=False,
            pretrained=True,
            optimizer="auto",
            patience=args.patience,
            seed=args.seed,
            deterministic=True,
            amp=dispositivo != "cpu",
            cache=False,
            plots=True,
            val=True,
            save=True,
            close_mosaic=10 if epochs > 10 else 0,
            freeze=args.freeze or None,
            fraction=fraction,
        )
    except torch.OutOfMemoryError as error:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("ERROR: memoria CUDA insuficiente.", file=sys.stderr)
        print("Reintenta con --batch 2; si persiste, agrega --imgsz 512.", file=sys.stderr)
        raise SystemExit(3) from error

    carpeta = Path(resultados.save_dir).resolve()
    configuracion["fin_utc"] = datetime.now(timezone.utc).isoformat()
    configuracion["directorio_resultados"] = str(carpeta)
    guardar_json(carpeta / "configuracion_reproducible.json", configuracion)

    print(f"Entrenamiento finalizado: {carpeta}")
    print(f"Mejores pesos: {carpeta / 'weights' / 'best.pt'}")
    print(f"Ultimos pesos: {carpeta / 'weights' / 'last.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
