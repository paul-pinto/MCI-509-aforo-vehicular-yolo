#!/usr/bin/env python3
"""Ejecuta inferencia YOLO sobre una imagen o carpeta y exporta resultados."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="Ruta a best.pt.")
    p.add_argument("--source", type=Path, required=True, help="Imagen o carpeta de imágenes.")
    p.add_argument("--output", type=Path, default=Path("resultados/predicciones"))
    p.add_argument("--name", default="predict")
    p.add_argument("--device", default="cpu", help="cpu o índice CUDA, por ejemplo 0.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.70)
    return p.parse_args()


def main() -> int:
    args = argumentos()
    modelo_path = args.model.resolve()
    fuente = args.source.resolve()
    salida = args.output.resolve()
    if not modelo_path.is_file():
        raise FileNotFoundError(f"No existe el modelo: {modelo_path}")
    if not fuente.exists():
        raise FileNotFoundError(f"No existe la fuente: {fuente}")
    if args.imgsz <= 0 or not 0 < args.conf < 1 or not 0 < args.iou <= 1:
        raise ValueError("imgsz debe ser positivo; conf e iou deben estar entre 0 y 1")

    import torch
    from ultralytics import YOLO

    if args.device.lower() == "cpu":
        checkpoint = torch.load(modelo_path, map_location="cpu", weights_only=False)
        del checkpoint

    modelo = YOLO(str(modelo_path))
    if args.device.lower() == "cpu":
        modelo.to("cpu")
        assert next(modelo.model.parameters()).device.type == "cpu"

    filas = []
    directorio_resultados = None
    resultados = modelo.predict(
        source=str(fuente),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        project=str(salida),
        name=args.name,
        exist_ok=False,
        save=True,
        stream=True,
        verbose=True,
    )
    for resultado in resultados:
        directorio_resultados = Path(resultado.save_dir).resolve()
        if resultado.boxes is None:
            continue
        for caja, clase, confianza in zip(
            resultado.boxes.xyxy.cpu().tolist(),
            resultado.boxes.cls.cpu().tolist(),
            resultado.boxes.conf.cpu().tolist(),
        ):
            filas.append({
                "imagen": Path(resultado.path).name,
                "clase_id": int(clase),
                "clase": modelo.names[int(clase)],
                "confianza": float(confianza),
                "x1": float(caja[0]),
                "y1": float(caja[1]),
                "x2": float(caja[2]),
                "y2": float(caja[3]),
            })

    if directorio_resultados is None:
        raise RuntimeError("Ultralytics no procesó ningún archivo compatible")
    columnas = ["imagen", "clase_id", "clase", "confianza", "x1", "y1", "x2", "y2"]
    with (directorio_resultados / "predicciones.csv").open("w", newline="", encoding="utf-8-sig") as archivo:
        w = csv.DictWriter(archivo, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)
    resumen = {
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "modelo": str(modelo_path),
        "fuente": str(fuente),
        "device": args.device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou_nms": args.iou,
        "detecciones": len(filas),
        "directorio_resultados": str(directorio_resultados),
    }
    (directorio_resultados / "resumen_prediccion.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Detecciones: {len(filas)}")
    print(f"Resultados: {directorio_resultados}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
