#!/usr/bin/env python3
"""Captura cuadros espaciados desde video/stream para adaptación de dominio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import time
from datetime import datetime
from pathlib import Path

def argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=os.getenv("CAMERA_URL", ""))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--session", required=True, help="Identificador, por ejemplo flujo_01.")
    p.add_argument("--source-id", default="youtube_NfsyRx50gAI")
    p.add_argument("--interval", type=float, default=5.0, help="Segundos entre capturas aceptadas.")
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--min-blur", type=float, default=25.0, help="Varianza Laplaciana mínima.")
    p.add_argument("--min-hamming", type=int, default=4, help="Distancia dHash mínima respecto al último cuadro.")
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--reconnect", type=float, default=3.0)
    p.add_argument("--max-failures", type=int, default=30)
    return p.parse_args()


def normalizar_fuente(valor: str) -> int | str:
    valor = valor.strip()
    if not valor:
        raise ValueError("La fuente está vacía. Renueve CAMERA_URL o use --source.")
    return int(valor) if valor.isdigit() else valor


def abrir_captura(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, int) and os.name == "nt":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    elif isinstance(source, str) and source.lower().startswith(("http://", "https://", "rtsp://", "rtsps://")):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def nitidez(frame) -> float:
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def dhash(frame) -> int:
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pequeno = cv2.resize(gris, (9, 8), interpolation=cv2.INTER_AREA)
    bits = pequeno[:, 1:] > pequeno[:, :-1]
    valor = 0
    for bit in bits.ravel():
        valor = (valor << 1) | int(bit)
    return valor


def distancia_hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def guardar_jpeg(ruta: Path, frame, calidad: int) -> None:
    ok, datos = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, calidad])
    if not ok:
        raise RuntimeError(f"No se pudo codificar {ruta}")
    datos.tofile(ruta)


def sha256(ruta: Path) -> str:
    d = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            d.update(bloque)
    return d.hexdigest()


def main() -> int:
    args = argumentos()
    global cv2, np
    import cv2
    import numpy as np

    if args.interval <= 0 or args.max_frames <= 0:
        raise ValueError("interval y max-frames deben ser positivos")
    if not 0 <= args.min_hamming <= 64 or not 1 <= args.jpeg_quality <= 100:
        raise ValueError("min-hamming debe estar entre 0 y 64 y jpeg-quality entre 1 y 100")
    source = normalizar_fuente(args.source)
    raiz = args.output.resolve() / args.session
    dir_imagenes = raiz / "images"
    dir_imagenes.mkdir(parents=True, exist_ok=True)
    manifiesto = raiz / "manifest.csv"
    existentes = sorted(dir_imagenes.glob("*.jpg"))
    guardados = len(existentes)
    if guardados >= args.max_frames:
        print(f"La sesión ya contiene {guardados} cuadros; no se capturó nada.")
        return 0

    nuevo_csv = not manifiesto.exists() or manifiesto.stat().st_size == 0
    archivo_csv = manifiesto.open("a", newline="", encoding="utf-8-sig")
    campos = ["session", "index", "filename", "timestamp_local", "source_id", "width", "height", "blur_score", "dhash", "sha256"]
    writer = csv.DictWriter(archivo_csv, fieldnames=campos)
    if nuevo_csv:
        writer.writeheader()

    cap = abrir_captura(source)
    if not cap.isOpened():
        archivo_csv.close()
        raise RuntimeError("No se pudo abrir la fuente")
    ultimo_hash = None
    if existentes:
        imagen = cv2.imdecode(np.fromfile(existentes[-1], dtype=np.uint8), cv2.IMREAD_COLOR)
        if imagen is not None:
            ultimo_hash = dhash(imagen)
    siguiente = time.monotonic()
    fallos = 0
    rechazados_blur = 0
    rechazados_similitud = 0

    try:
        while guardados < args.max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                fallos += 1
                if fallos >= args.max_failures:
                    print(f"Stream interrumpido; reconectando en {args.reconnect:.1f} s...")
                    cap.release(); time.sleep(args.reconnect); cap = abrir_captura(source); fallos = 0
                continue
            fallos = 0
            ahora = time.monotonic()
            if ahora < siguiente:
                if not args.no_display:
                    vista = frame.copy()
                    cv2.putText(vista, f"Capturados {guardados}/{args.max_frames}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 0), 2)
                    cv2.imshow("Captura de dataset | Q para terminar", vista)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
                continue

            score = nitidez(frame)
            hash_actual = dhash(frame)
            if score < args.min_blur:
                rechazados_blur += 1; siguiente = ahora + min(1.0, args.interval); continue
            if ultimo_hash is not None and distancia_hamming(hash_actual, ultimo_hash) < args.min_hamming:
                rechazados_similitud += 1; siguiente = ahora + min(1.0, args.interval); continue

            indice = guardados + 1
            timestamp = datetime.now().astimezone()
            nombre = f"{args.session}_{indice:04d}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            ruta = dir_imagenes / nombre
            guardar_jpeg(ruta, frame, args.jpeg_quality)
            alto, ancho = frame.shape[:2]
            writer.writerow({
                "session": args.session, "index": indice, "filename": nombre,
                "timestamp_local": timestamp.isoformat(timespec="seconds"), "source_id": args.source_id,
                "width": ancho, "height": alto, "blur_score": f"{score:.4f}",
                "dhash": f"{hash_actual:016x}", "sha256": sha256(ruta),
            })
            archivo_csv.flush()
            guardados += 1; ultimo_hash = hash_actual; siguiente = ahora + args.interval
            print(f"[{guardados:03d}/{args.max_frames}] {nombre} | nitidez={score:.1f}")
    finally:
        cap.release(); archivo_csv.close(); cv2.destroyAllWindows()

    print(f"Capturados: {guardados}")
    print(f"Rechazados por desenfoque: {rechazados_blur}")
    print(f"Rechazados por similitud: {rechazados_similitud}")
    print(f"Sesión: {raiz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
