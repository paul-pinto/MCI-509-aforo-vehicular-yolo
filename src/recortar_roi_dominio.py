#!/usr/bin/env python3
"""Recorta una ROI y transforma pseudoetiquetas YOLO conservando trazabilidad."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def read_image(path: Path):
    """Lee imágenes desde rutas Unicode en Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image) -> None:
    """Escribe imágenes en rutas Unicode en Windows."""
    extension = path.suffix.lower()
    if extension == ".jpeg":
        extension = ".jpg"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"No se pudo codificar la imagen: {path}")
    encoded.tofile(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recorta una ROI de un dataset YOLO")
    parser.add_argument("--input", required=True, help="Raíz con images/ y labels/")
    parser.add_argument("--output", required=True, help="Directorio de salida")
    parser.add_argument("--x1", type=int)
    parser.add_argument("--y1", type=int)
    parser.add_argument("--x2", type=int)
    parser.add_argument("--y2", type=int)
    parser.add_argument(
        "--min-retained",
        type=float,
        default=0.60,
        help="Fracción mínima de la caja original dentro de la ROI",
    )
    parser.add_argument("--preview-limit", type=int, default=20)
    return parser.parse_args()


def choose_roi(image, image_name: str) -> tuple[int, int, int, int]:
    title = "Seleccione ROI y presione ENTER o ESPACIO; C cancela"
    x, y, w, h = cv2.selectROI(title, image, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Selección cancelada para {image_name}")
    return int(x), int(y), int(x + w), int(y + h)


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.is_file():
        return []
    labels = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Etiqueta inválida {path}:{line_number}")
        cls_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])
        labels.append((cls_id, xc, yc, bw, bh))
    return labels


def transform_box(
    label: tuple[int, float, float, float, float],
    image_w: int,
    image_h: int,
    roi: tuple[int, int, int, int],
    min_retained: float,
) -> tuple[int, float, float, float, float] | None:
    cls_id, xc, yc, bw, bh = label
    bx1 = (xc - bw / 2) * image_w
    by1 = (yc - bh / 2) * image_h
    bx2 = (xc + bw / 2) * image_w
    by2 = (yc + bh / 2) * image_h
    original_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if original_area <= 0:
        return None

    rx1, ry1, rx2, ry2 = roi
    cx1, cy1 = max(bx1, rx1), max(by1, ry1)
    cx2, cy2 = min(bx2, rx2), min(by2, ry2)
    clipped_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    if clipped_area / original_area < min_retained:
        return None

    roi_w, roi_h = rx2 - rx1, ry2 - ry1
    nx1, ny1 = cx1 - rx1, cy1 - ry1
    nx2, ny2 = cx2 - rx1, cy2 - ry1
    nxc = ((nx1 + nx2) / 2) / roi_w
    nyc = ((ny1 + ny2) / 2) / roi_h
    nbw = (nx2 - nx1) / roi_w
    nbh = (ny2 - ny1) / roi_h
    return cls_id, nxc, nyc, nbw, nbh


def main() -> int:
    args = parse_args()
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    images_dir = input_root / "images"
    labels_dir = input_root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("La entrada debe contener images/ y labels/")
    if not 0 < args.min_retained <= 1:
        raise ValueError("--min-retained debe estar en (0, 1]")

    images = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not images:
        raise RuntimeError("No se encontraron imágenes")

    first = read_image(images[0])
    if first is None:
        raise RuntimeError(f"No se pudo leer {images[0]}")
    first_h, first_w = first.shape[:2]

    coordinates = (args.x1, args.y1, args.x2, args.y2)
    if all(value is not None for value in coordinates):
        roi = tuple(int(value) for value in coordinates)
    elif all(value is None for value in coordinates):
        roi = choose_roi(first, images[0].name)
    else:
        raise ValueError("Indique las cuatro coordenadas o ninguna")

    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= first_w and 0 <= y1 < y2 <= first_h):
        raise ValueError(f"ROI fuera de la imagen {first_w}x{first_h}: {roi}")

    output_images = output_root / "images"
    output_labels = output_root / "labels"
    output_previews = output_root / "previews"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    if args.preview_limit > 0:
        output_previews.mkdir(parents=True, exist_ok=True)

    kept_boxes = 0
    discarded_boxes = 0
    empty_images = 0

    for index, image_path in enumerate(images, 1):
        image = read_image(image_path)
        if image is None:
            raise RuntimeError(f"No se pudo leer {image_path}")
        height, width = image.shape[:2]
        if (width, height) != (first_w, first_h):
            raise ValueError(f"Resolución inconsistente: {image_path} -> {width}x{height}")

        crop = image[y1:y2, x1:x2].copy()
        transformed = []
        original_labels = read_labels(labels_dir / f"{image_path.stem}.txt")
        for label in original_labels:
            converted = transform_box(label, width, height, roi, args.min_retained)
            if converted is None:
                discarded_boxes += 1
            else:
                transformed.append(converted)
                kept_boxes += 1

        write_image(output_images / image_path.name, crop)
        lines = [
            f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            for cls_id, xc, yc, bw, bh in transformed
        ]
        (output_labels / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        if not transformed:
            empty_images += 1

        if index <= args.preview_limit:
            preview = crop.copy()
            crop_h, crop_w = preview.shape[:2]
            for cls_id, xc, yc, bw, bh in transformed:
                px1 = int((xc - bw / 2) * crop_w)
                py1 = int((yc - bh / 2) * crop_h)
                px2 = int((xc + bw / 2) * crop_w)
                py2 = int((yc + bh / 2) * crop_h)
                cv2.rectangle(preview, (px1, py1), (px2, py2), (0, 220, 255), 2)
                cv2.putText(preview, str(cls_id), (px1, max(16, py1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
            write_image(output_previews / image_path.name, preview)

        print(f"[{index:03d}/{len(images):03d}] {image_path.name}: {len(transformed)} cajas")

    classes_path = input_root / "classes.txt"
    if classes_path.is_file():
        shutil.copy2(classes_path, output_root / "classes.txt")

    summary = {
        "input": str(input_root),
        "images": len(images),
        "original_resolution": [first_w, first_h],
        "roi_xyxy": [x1, y1, x2, y2],
        "roi_resolution": [x2 - x1, y2 - y1],
        "min_retained_fraction": args.min_retained,
        "kept_boxes": kept_boxes,
        "discarded_boxes": discarded_boxes,
        "images_without_boxes": empty_images,
    }
    (output_root / "roi_config.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("\nROI preparada")
    print(f"Coordenadas xyxy: {roi}")
    print(f"Resolución ROI: {x2 - x1}x{y2 - y1}")
    print(f"Cajas conservadas: {kept_boxes}")
    print(f"Cajas descartadas: {discarded_boxes}")
    print(f"Imágenes sin cajas: {empty_images}")
    print(f"Salida: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

