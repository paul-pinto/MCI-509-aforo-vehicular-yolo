#!/usr/bin/env python3
"""Genera ejemplos correctos y errores del split test con criterios objetivos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--classes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou-match", type=float, default=0.50)
    p.add_argument("--device", default="0")
    p.add_argument("--correctos", type=int, default=8)
    p.add_argument("--errores", type=int, default=7)
    return p.parse_args()


def leer_imagen(ruta: Path):
    datos = np.fromfile(ruta, dtype=np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if imagen is None:
        raise ValueError(f"No se pudo leer {ruta}")
    return imagen


def guardar_imagen(ruta: Path, imagen) -> None:
    extension = ruta.suffix or ".jpg"
    ok, codificada = cv2.imencode(extension, imagen)
    if not ok:
        raise ValueError(f"No se pudo codificar {ruta}")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    codificada.tofile(ruta)


def leer_gt(ruta: Path, ancho: int, alto: int) -> list[dict]:
    cajas = []
    for linea in ruta.read_text(encoding="utf-8-sig").splitlines():
        if not linea.strip():
            continue
        clase, x, y, w, h = map(float, linea.split())
        cajas.append({
            "clase": int(clase),
            "xyxy": [
                (x - w / 2) * ancho,
                (y - h / 2) * alto,
                (x + w / 2) * ancho,
                (y + h / 2) * alto,
            ],
        })
    return cajas


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def emparejar(gt: list[dict], pred: list[dict], umbral: float):
    candidatos = []
    for ig, g in enumerate(gt):
        for ip, p in enumerate(pred):
            if g["clase"] == p["clase"]:
                candidatos.append((iou(g["xyxy"], p["xyxy"]), ig, ip))
    usados_gt, usados_pred, matches = set(), set(), []
    for valor, ig, ip in sorted(candidatos, reverse=True):
        if valor >= umbral and ig not in usados_gt and ip not in usados_pred:
            usados_gt.add(ig)
            usados_pred.add(ip)
            matches.append((ig, ip, valor))
    return matches, usados_gt, usados_pred


def dibujar_caja(imagen, caja, color, texto, grosor=2):
    x1, y1, x2, y2 = (int(round(x)) for x in caja)
    cv2.rectangle(imagen, (x1, y1), (x2, y2), color, grosor)
    escala = max(0.45, min(imagen.shape[:2]) / 1200)
    (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)
    y_texto = max(th + 4, y1)
    cv2.rectangle(imagen, (x1, y_texto - th - 4), (x1 + tw + 4, y_texto + base), color, -1)
    cv2.putText(imagen, texto, (x1 + 2, y_texto - 2), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 1, cv2.LINE_AA)


def renderizar(item: dict, nombres: list[str], destino: Path) -> None:
    imagen = leer_imagen(Path(item["imagen"]))
    matched_gt = {m[0] for m in item["matches"]}
    matched_pred = {m[1] for m in item["matches"]}
    for indice, g in enumerate(item["gt"]):
        if indice not in matched_gt:
            dibujar_caja(imagen, g["xyxy"], (0, 165, 255), f"FN GT:{nombres[g['clase']]}", 3)
    for indice, p in enumerate(item["pred"]):
        if indice in matched_pred:
            dibujar_caja(imagen, p["xyxy"], (255, 100, 0), f"TP {nombres[p['clase']]} {p['conf']:.2f}")
        else:
            dibujar_caja(imagen, p["xyxy"], (0, 0, 255), f"FP {nombres[p['clase']]} {p['conf']:.2f}", 3)
    titulo = f"TP={item['tp']}  FP={item['fp']}  FN={item['fn']}  IoU medio={item['iou_medio']:.3f}"
    cv2.rectangle(imagen, (0, 0), (min(imagen.shape[1], 620), 34), (20, 20, 20), -1)
    cv2.putText(imagen, titulo, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    guardar_imagen(destino, imagen)


def main() -> int:
    args = argumentos()
    global cv2, np
    import cv2
    import numpy as np

    modelo_path, imagenes, etiquetas = args.model.resolve(), args.images.resolve(), args.labels.resolve()
    salida = args.output.resolve()
    for ruta, descripcion in ((modelo_path, "modelo"), (imagenes, "imagenes"), (etiquetas, "etiquetas"), (args.classes.resolve(), "clases")):
        if not ruta.exists():
            raise FileNotFoundError(f"No existe {descripcion}: {ruta}")
    if not 0 < args.conf < 1 or not 0 < args.iou_match <= 1:
        raise ValueError("conf e iou-match deben estar entre 0 y 1")
    nombres = [x.strip() for x in args.classes.resolve().read_text(encoding="utf-8-sig").splitlines() if x.strip()]

    from ultralytics import YOLO
    modelo = YOLO(str(modelo_path))
    resultados = modelo.predict(
        source=str(imagenes), imgsz=args.imgsz, conf=args.conf, device=args.device,
        stream=True, save=False, verbose=False,
    )

    analisis = []
    for resultado in resultados:
        ruta_imagen = Path(resultado.path)
        alto, ancho = resultado.orig_shape
        ruta_etiqueta = etiquetas / f"{ruta_imagen.stem}.txt"
        if not ruta_etiqueta.exists():
            raise FileNotFoundError(f"Falta etiqueta test: {ruta_etiqueta}")
        gt = leer_gt(ruta_etiqueta, ancho, alto)
        pred = []
        if resultado.boxes is not None:
            for caja, clase, confianza in zip(
                resultado.boxes.xyxy.cpu().tolist(),
                resultado.boxes.cls.cpu().tolist(),
                resultado.boxes.conf.cpu().tolist(),
            ):
                pred.append({"clase": int(clase), "xyxy": caja, "conf": float(confianza)})
        matches, usados_gt, usados_pred = emparejar(gt, pred, args.iou_match)
        valores_iou = [m[2] for m in matches]
        tp, fp, fn = len(matches), len(pred) - len(usados_pred), len(gt) - len(usados_gt)
        analisis.append({
            "imagen": str(ruta_imagen), "gt": gt, "pred": pred, "matches": matches,
            "tp": tp, "fp": fp, "fn": fn,
            "iou_medio": sum(valores_iou) / len(valores_iou) if valores_iou else 0.0,
            "conf_media": sum(p["conf"] for p in pred) / len(pred) if pred else 0.0,
        })

    correctos = [x for x in analisis if x["fp"] == 0 and x["fn"] == 0 and x["tp"] > 0]
    correctos.sort(key=lambda x: (x["iou_medio"], x["conf_media"], x["tp"]), reverse=True)
    errores = [x for x in analisis if x["fp"] + x["fn"] > 0]
    errores.sort(key=lambda x: (x["fp"] + x["fn"], x["fn"], 1 - x["iou_medio"]), reverse=True)
    seleccion = [("correcto", x) for x in correctos[:args.correctos]] + [("error", x) for x in errores[:args.errores]]

    filas_todas = []
    for x in analisis:
        filas_todas.append({
            "imagen": Path(x["imagen"]).name, "tp": x["tp"], "fp": x["fp"], "fn": x["fn"],
            "iou_medio": round(x["iou_medio"], 6), "conf_media": round(x["conf_media"], 6),
            "categoria": "correcto" if x["fp"] == 0 and x["fn"] == 0 else "error",
        })
    salida.mkdir(parents=True, exist_ok=True)
    with (salida / "analisis_predicciones_test.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_todas[0].keys())); w.writeheader(); w.writerows(filas_todas)

    filas_seleccion = []
    for numero, (categoria, item) in enumerate(seleccion, 1):
        nombre = f"{numero:02d}_{Path(item['imagen']).stem}.jpg"
        destino = salida / ("correctos" if categoria == "correcto" else "errores") / nombre
        renderizar(item, nombres, destino)
        filas_seleccion.append({
            "numero": numero, "categoria": categoria, "archivo_salida": str(destino.relative_to(salida)),
            "imagen_original": item["imagen"], "tp": item["tp"], "fp": item["fp"], "fn": item["fn"],
            "iou_medio": round(item["iou_medio"], 6), "conf_media": round(item["conf_media"], 6),
        })
    with (salida / "seleccion_evidencia.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_seleccion[0].keys())); w.writeheader(); w.writerows(filas_seleccion)
    resumen = {
        "imagenes_analizadas": len(analisis), "conf_prediccion": args.conf, "iou_match": args.iou_match,
        "imagenes_sin_error": len(correctos), "imagenes_con_error": len(errores),
        "ejemplos_correctos_guardados": min(args.correctos, len(correctos)),
        "ejemplos_error_guardados": min(args.errores, len(errores)),
        "nota": "Analisis posterior a la evaluacion final; no usar para ajustar el modelo.",
    }
    (salida / "resumen_evidencia.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"Resultados: {salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
