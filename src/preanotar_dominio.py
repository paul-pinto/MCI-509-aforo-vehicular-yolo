#!/usr/bin/env python3
"""Preanota fotogramas del dominio objetivo con YOLO-COCO.

Convierte las clases COCO vehiculares al esquema del proyecto:
car->car, motorcycle->motorbike, bus->bus y truck->truck.
Las clases threewheel y van requieren revisión/anotación manual.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_NAMES = {
    0: "car",
    1: "threewheel",
    2: "bus",
    3: "truck",
    4: "motorbike",
    5: "van",
}

# Índices COCO: car=2, motorcycle=3, bus=5, truck=7.
COCO_TO_PROJECT = {2: 0, 3: 4, 5: 2, 7: 3}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera pseudoetiquetas YOLO para revisión manual."
    )
    parser.add_argument("--source", required=True, help="Directorio con imágenes")
    parser.add_argument("--output", required=True, help="Directorio de salida")
    parser.add_argument("--model", default="yolo11n.pt", help="Modelo profesor COCO")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=100,
        help="Cantidad máxima de imágenes de vista previa; 0 desactiva",
    )
    return parser.parse_args()


def yolo_line(cls_id: int, xywhn: list[float]) -> str:
    values = " ".join(f"{value:.6f}" for value in xywhn)
    return f"{cls_id} {values}"


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"No existe el directorio de entrada: {source}")

    images = sorted(
        path for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No se encontraron imágenes en: {source}")

    images_out = output / "images"
    labels_out = output / "labels"
    previews_out = output / "previews"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    if args.preview_limit > 0:
        previews_out.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    empty_images = 0

    print(f"Imágenes: {len(images)}")
    print(f"Modelo profesor: {args.model}")
    print(f"Salida: {output}")

    for index, image_path in enumerate(images, start=1):
        destination = images_out / image_path.name
        if image_path != destination:
            shutil.copy2(image_path, destination)

        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=sorted(COCO_TO_PROJECT),
            device=args.device,
            max_det=args.max_det,
            verbose=False,
        )[0]

        label_lines: list[str] = []
        preview = cv2.imread(str(image_path)) if index <= args.preview_limit else None

        if result.boxes is not None:
            for box in result.boxes:
                coco_id = int(box.cls.item())
                project_id = COCO_TO_PROJECT[coco_id]
                confidence = float(box.conf.item())
                xywhn = [float(value) for value in box.xywhn[0].tolist()]
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                class_name = PROJECT_NAMES[project_id]

                label_lines.append(yolo_line(project_id, xywhn))
                counts[class_name] += 1
                rows.append(
                    {
                        "imagen": image_path.name,
                        "clase_coco_id": coco_id,
                        "clase_proyecto_id": project_id,
                        "clase_proyecto": class_name,
                        "confianza": f"{confidence:.6f}",
                        "x_centro_norm": f"{xywhn[0]:.6f}",
                        "y_centro_norm": f"{xywhn[1]:.6f}",
                        "ancho_norm": f"{xywhn[2]:.6f}",
                        "alto_norm": f"{xywhn[3]:.6f}",
                        "requiere_revision": "si",
                    }
                )

                if preview is not None:
                    x1, y1, x2, y2 = (int(round(value)) for value in xyxy)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 255), 2)
                    cv2.putText(
                        preview,
                        f"{class_name} {confidence:.2f}",
                        (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )

        (labels_out / f"{image_path.stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        if not label_lines:
            empty_images += 1

        if preview is not None:
            cv2.imwrite(str(previews_out / image_path.name), preview)

        print(
            f"[{index:03d}/{len(images):03d}] {image_path.name}: "
            f"{len(label_lines)} detecciones"
        )

    csv_path = output / "pseudoetiquetas_auditoria.csv"
    fieldnames = [
        "imagen", "clase_coco_id", "clase_proyecto_id", "clase_proyecto",
        "confianza", "x_centro_norm", "y_centro_norm", "ancho_norm",
        "alto_norm", "requiere_revision",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (output / "classes.txt").write_text(
        "\n".join(PROJECT_NAMES[index] for index in sorted(PROJECT_NAMES)) + "\n",
        encoding="utf-8",
    )
    summary = {
        "source": str(source),
        "teacher_model": args.model,
        "imgsz": args.imgsz,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "images": len(images),
        "images_without_detections": empty_images,
        "pseudo_boxes": len(rows),
        "boxes_by_class": dict(sorted(counts.items())),
        "manual_review_required": True,
        "manual_only_classes": ["threewheel", "van"],
    }
    (output / "resumen_preanotacion.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nPreanotación terminada")
    print(f"Pseudoetiquetas: {len(rows)}")
    print(f"Imágenes sin detecciones: {empty_images}")
    print(f"Por clase: {dict(sorted(counts.items()))}")
    print("IMPORTANTE: todas las etiquetas deben revisarse manualmente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
