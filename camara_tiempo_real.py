#!/usr/bin/env python3
"""Detección y tracking de personas/vehículos con YOLO y ByteTrack.

Fuentes compatibles:
  - Webcam local: --source 0
  - RTSP/HTTP/HLS: --source "$env:CAMERA_URL" (PowerShell)
  - Archivo local: --source video.mp4

Pulsa Q o Esc para cerrar la ventana.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import torch
from ultralytics import YOLO


CLASES_TRAFICO_COCO = [0, 1, 2, 3, 5, 7]
NOMBRES_TRAFICO_COCO = {0: "persona", 1: "bicicleta", 2: "auto", 3: "moto", 5: "bus", 7: "camion"}
CLASES_MODELO_VEHICULAR = {"car", "threewheel", "bus", "truck", "motorbike", "van"}
TRADUCCION_MODELO_VEHICULAR = {
    "car": "auto",
    "threewheel": "triciclo",
    "bus": "bus",
    "truck": "camion",
    "motorbike": "moto",
    "van": "furgoneta",
}


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO + ByteTrack sobre webcam, cámara IP o video."
    )
    parser.add_argument(
        "--source",
        default=os.getenv("CAMERA_URL", "0"),
        help="Índice de webcam, URL RTSP/HTTP/HLS o ruta de video. "
        "Por defecto usa CAMERA_URL y, si no existe, la webcam 0.",
    )
    parser.add_argument("--model", default="yolo11n.pt", help="Modelo de Ultralytics.")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0],
        help="IDs del modelo. Con --traffic se seleccionan automáticamente.",
    )
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--width", type=int, default=1280, help="Ancho solicitado a webcam.")
    parser.add_argument("--height", type=int, default=720, help="Alto solicitado a webcam.")
    parser.add_argument("--skip", type=int, default=0, help="Frames omitidos entre inferencias.")
    parser.add_argument("--reconnect", type=float, default=3.0, help="Espera al reconectar stream.")
    parser.add_argument("--max-failures", type=int, default=30)
    parser.add_argument("--record", type=Path, help="Guardar salida anotada en MP4.")
    parser.add_argument("--no-display", action="store_true", help="Procesar sin abrir ventana.")
    parser.add_argument(
        "--traffic",
        action="store_true",
        help="Usar automáticamente las clases de tráfico de COCO o del modelo fine-tuned.",
    )
    parser.add_argument("--setup", action="store_true", help="Marcar ROI y línea con el mouse.")
    parser.add_argument("--config", type=Path, default=Path("config_trafico.json"))
    parser.add_argument("--events", type=Path, default=Path("eventos_trafico.csv"))
    parser.add_argument("--no-analytics", action="store_true", help="Desactivar ROI y conteo.")
    return parser.parse_args()


def seleccionar_puntos(frame, titulo: str, minimo: int, maximo: int | None = None):
    """Recoge puntos con clic izquierdo; Enter confirma y R reinicia."""
    puntos: list[tuple[int, int]] = []

    def mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and (maximo is None or len(puntos) < maximo):
            puntos.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and puntos:
            puntos.pop()

    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(titulo, mouse)
    while True:
        vista = frame.copy()
        for i, punto in enumerate(puntos):
            cv2.circle(vista, punto, 6, (0, 255, 255), -1)
            if i:
                cv2.line(vista, puntos[i - 1], punto, (0, 255, 255), 2)
        if maximo is None and len(puntos) > 2:
            cv2.line(vista, puntos[-1], puntos[0], (0, 255, 255), 2)
        ayuda = "Clic: agregar | clic derecho: deshacer | R: reiniciar | Enter: confirmar"
        cv2.rectangle(vista, (5, 5), (900, 42), (0, 0, 0), -1)
        cv2.putText(vista, ayuda, (15, 31), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        cv2.imshow(titulo, vista)
        tecla = cv2.waitKey(20) & 0xFF
        if tecla in (13, 10) and len(puntos) >= minimo:
            break
        if tecla in (ord("r"), ord("R")):
            puntos.clear()
        if tecla == 27:
            raise KeyboardInterrupt
    cv2.destroyWindow(titulo)
    return puntos


def configurar_analitica(frame, ruta: Path) -> dict:
    roi = seleccionar_puntos(frame, "1/2 Marca la calzada (ROI)", 3)
    linea = seleccionar_puntos(frame, "2/2 Marca la linea de conteo", 2, 2)
    alto, ancho = frame.shape[:2]
    datos = {
        "frame_size": [ancho, alto],
        "roi": [[x / ancho, y / alto] for x, y in roi],
        "line": [[x / ancho, y / alto] for x, y in linea],
        "direction_labels": ["A", "B"],
    }
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2), encoding="utf-8")
    print(f"Configuración guardada en: {ruta.resolve()}")
    return datos


def cargar_analitica(ruta: Path, frame) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[str]]:
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    alto, ancho = frame.shape[:2]
    escalar = lambda p: (round(p[0] * ancho), round(p[1] * alto))
    return [escalar(p) for p in datos["roi"]], [escalar(p) for p in datos["line"]], datos.get("direction_labels", ["A", "B"])


def lado_linea(punto, a, b) -> float:
    return (b[0] - a[0]) * (punto[1] - a[1]) - (b[1] - a[1]) * (punto[0] - a[0])


def abrir_eventos(ruta: Path):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not ruta.exists() or ruta.stat().st_size == 0
    archivo = ruta.open("a", newline="", encoding="utf-8")
    writer = csv.writer(archivo)
    if nuevo:
        writer.writerow(["timestamp", "track_id", "clase_id", "clase", "direccion", "confianza"])
        archivo.flush()
    return archivo, writer


def normalizar_fuente(valor: str) -> int | str:
    return int(valor) if valor.strip().isdigit() else valor.strip()


def es_stream_remoto(source: int | str) -> bool:
    return isinstance(source, str) and source.lower().startswith(
        ("rtsp://", "rtsps://", "http://", "https://")
    )


def fuente_segura(source: int | str) -> str:
    """Oculta usuario y contraseña al mostrar una URL."""
    if not es_stream_remoto(source):
        return str(source)
    partes = urlsplit(str(source))
    host = partes.hostname or ""
    if partes.port:
        host += f":{partes.port}"
    netloc = f"***:***@{host}" if partes.username or partes.password else host
    return urlunsplit((partes.scheme, netloc, partes.path, partes.query, ""))


def abrir_captura(source: int | str, width: int, height: int) -> cv2.VideoCapture:
    if isinstance(source, int) and os.name == "nt":
        captura = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    elif es_stream_remoto(source):
        captura = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        captura = cv2.VideoCapture(source)

    captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if isinstance(source, int):
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return captura


def abrir_writer(ruta: Path, fps: float, ancho: int, alto: int) -> cv2.VideoWriter:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (ancho, alto)
    )
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el video de salida: {ruta}")
    return writer


def configurar_clases_modelo(modelo: YOLO, args: argparse.Namespace) -> tuple[list[int], dict[int, str], str]:
    nombres_originales = {int(k): str(v) for k, v in modelo.names.items()}
    nombres_normalizados = {k: v.strip().lower() for k, v in nombres_originales.items()}
    es_modelo_vehicular = CLASES_MODELO_VEHICULAR.issubset(set(nombres_normalizados.values()))

    if args.traffic and es_modelo_vehicular:
        clases = [k for k, v in nombres_normalizados.items() if v in CLASES_MODELO_VEHICULAR]
        nombres = {k: TRADUCCION_MODELO_VEHICULAR[nombres_normalizados[k]] for k in clases}
        perfil = "vehicular_fine_tuned"
    elif args.traffic:
        clases = [k for k in CLASES_TRAFICO_COCO if k in nombres_originales]
        nombres = {k: NOMBRES_TRAFICO_COCO.get(k, nombres_originales[k]) for k in clases}
        perfil = "coco_trafico"
    else:
        clases = args.classes
        invalidas = [k for k in clases if k not in nombres_originales]
        if invalidas:
            raise ValueError(
                f"Clases inexistentes en el modelo: {invalidas}. "
                f"IDs disponibles: {sorted(nombres_originales)}"
            )
        nombres = {
            k: TRADUCCION_MODELO_VEHICULAR.get(nombres_normalizados[k], nombres_originales[k])
            for k in clases
        }
        perfil = "seleccion_manual"
    if not clases:
        raise ValueError("No se seleccionó ninguna clase compatible con el modelo")
    return sorted(clases), nombres, perfil


def main() -> int:
    args = argumentos()
    source = normalizar_fuente(args.source)
    modelo = YOLO(args.model)
    classes, nombres_clases, perfil_clases = configurar_clases_modelo(modelo, args)

    print(f"Fuente: {fuente_segura(source)}")
    print(f"Dispositivo: {args.device} | Modelo: {args.model}")
    print(f"Perfil de clases: {perfil_clases} | Clases: {[(i, nombres_clases[i]) for i in classes]}")
    if torch.cuda.is_available() and args.device != "cpu":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    captura = abrir_captura(source, args.width, args.height)
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {fuente_segura(source)}")

    writer: cv2.VideoWriter | None = None
    tiempos: deque[float] = deque(maxlen=30)
    fallos = 0
    frame_num = 0
    roi: list[tuple[int, int]] | None = None
    linea: list[tuple[int, int]] | None = None
    etiquetas_direccion = ["A", "B"]
    lados_previos: dict[int, float] = {}
    contados: set[int] = set()
    totales_direccion: Counter[str] = Counter()
    totales_clase: Counter[str] = Counter()
    eventos_archivo = None
    eventos_writer = None

    try:
        while True:
            ok, frame = captura.read()
            if not ok or frame is None:
                fallos += 1
                if not es_stream_remoto(source) or fallos >= args.max_failures:
                    if es_stream_remoto(source):
                        print(f"Flujo perdido. Reintentando en {args.reconnect:.1f} s...")
                        captura.release()
                        time.sleep(args.reconnect)
                        captura = abrir_captura(source, args.width, args.height)
                        fallos = 0
                        if captura.isOpened():
                            continue
                    break
                time.sleep(0.05)
                continue

            fallos = 0
            if not args.no_analytics and roi is None:
                if args.setup or not args.config.exists():
                    if args.no_display:
                        raise RuntimeError("--setup necesita una ventana; quite --no-display.")
                    configurar_analitica(frame, args.config)
                roi, linea, etiquetas_direccion = cargar_analitica(args.config, frame)
                eventos_archivo, eventos_writer = abrir_eventos(args.events)
                print(f"Analítica activa | ROI: {len(roi)} puntos | Línea: {linea}")

            frame_num += 1
            if args.skip and frame_num % (args.skip + 1) != 1:
                continue

            frame_ia = frame
            if roi is not None:
                mascara = frame.copy()
                mascara[:] = 0
                poligono = np.array(roi, dtype="int32")
                cv2.fillPoly(mascara, [poligono], (255, 255, 255))
                frame_ia = cv2.bitwise_and(frame, mascara)

            inicio = time.perf_counter()
            resultado = modelo.track(
                source=frame_ia,
                device=args.device,
                conf=args.conf,
                iou=args.iou,
                classes=classes,
                imgsz=args.imgsz,
                persist=True,
                tracker=args.tracker,
                verbose=False,
            )[0]
            tiempos.append(time.perf_counter() - inicio)

            anotado = frame.copy()
            fps_inferencia = len(tiempos) / sum(tiempos) if sum(tiempos) else 0.0
            cajas = resultado.boxes
            cantidad = len(cajas) if cajas is not None else 0
            ids = set()
            if cajas is not None:
                xyxy = cajas.xyxy.detach().cpu().tolist()
                clases = cajas.cls.detach().cpu().tolist()
                confianzas = cajas.conf.detach().cpu().tolist()
                track_ids = cajas.id.detach().cpu().tolist() if cajas.id is not None else [None] * len(xyxy)

                for box, clase_f, confianza, track_f in zip(xyxy, clases, confianzas, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    clase_id = int(clase_f)
                    track_id = int(track_f) if track_f is not None else None
                    centro = ((x1 + x2) // 2, y2)
                    if roi is not None and cv2.pointPolygonTest(poligono, centro, False) < 0:
                        continue
                    if track_id is not None:
                        ids.add(track_id)

                    nombre = nombres_clases.get(clase_id, str(modelo.names.get(clase_id, clase_id)))
                    color = (0, 220, 255)
                    cv2.rectangle(anotado, (x1, y1), (x2, y2), color, 2)
                    rotulo = f"{nombre} {confianza:.2f}" + (f" ID:{track_id}" if track_id is not None else "")
                    cv2.putText(anotado, rotulo, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, .52, color, 2, cv2.LINE_AA)
                    cv2.circle(anotado, centro, 4, color, -1)

                    if track_id is None or linea is None:
                        continue
                    lado = lado_linea(centro, linea[0], linea[1])
                    anterior = lados_previos.get(track_id)
                    # La zona muerta evita contar oscilaciones exactamente sobre la línea.
                    umbral = max(frame.shape[:2]) * 0.004
                    if abs(lado) > umbral:
                        if anterior is not None and anterior * lado < 0 and track_id not in contados:
                            direccion = etiquetas_direccion[0] if anterior < 0 < lado else etiquetas_direccion[1]
                            contados.add(track_id)
                            totales_direccion[direccion] += 1
                            totales_clase[nombre] += 1
                            eventos_writer.writerow([
                                datetime.now().astimezone().isoformat(timespec="seconds"),
                                track_id, clase_id, nombre, direccion, f"{confianza:.4f}",
                            ])
                            eventos_archivo.flush()
                            print(f"CRUCE | ID {track_id} | {nombre} | dirección {direccion}")
                        lados_previos[track_id] = lado

            if roi is not None:
                cv2.polylines(anotado, [poligono], True, (0, 255, 255), 2, cv2.LINE_AA)
            if linea is not None:
                cv2.line(anotado, linea[0], linea[1], (255, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(anotado, etiquetas_direccion[0], linea[0], cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 255), 2)
                cv2.putText(anotado, etiquetas_direccion[1], linea[1], cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 255), 2)

            texto = f"FPS IA: {fps_inferencia:.1f} | Objetos: {cantidad} | IDs visibles: {len(ids)}"
            cv2.rectangle(anotado, (8, 8), (690, 45), (0, 0, 0), -1)
            cv2.putText(
                anotado, texto, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (0, 255, 0), 2, cv2.LINE_AA,
            )

            if roi is not None:
                resumen_dir = " | ".join(f"{k}: {totales_direccion[k]}" for k in etiquetas_direccion)
                resumen_cls = " | ".join(f"{k}: {v}" for k, v in totales_clase.most_common()) or "Sin cruces"
                cv2.rectangle(anotado, (8, 52), (min(anotado.shape[1] - 8, 900), 112), (0, 0, 0), -1)
                cv2.putText(anotado, f"Cruces {resumen_dir}", (18, 77), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 255, 255), 2)
                cv2.putText(anotado, resumen_cls, (18, 103), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)

            if args.record:
                if writer is None:
                    alto, ancho = anotado.shape[:2]
                    fps_origen = captura.get(cv2.CAP_PROP_FPS)
                    fps_salida = fps_origen / (args.skip + 1) if fps_origen > 0 else 20.0
                    writer = abrir_writer(args.record, fps_salida, ancho, alto)
                writer.write(anotado)

            if not args.no_display:
                cv2.imshow("YOLO + ByteTrack | Q o Esc para salir", anotado)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
    finally:
        captura.release()
        if writer is not None:
            writer.release()
        if eventos_archivo is not None:
            eventos_archivo.close()
        cv2.destroyAllWindows()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Proceso terminado.")
    if args.record:
        print(f"Video guardado en: {args.record.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
