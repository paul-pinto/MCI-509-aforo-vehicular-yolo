import os
import threading
import time

from collections import deque

import cv2


class VideoService:
    def __init__(
        self,
        model_service,
        stream_service,
        traffic_config_service,
        traffic_counter,
    ):
        self.model_service = model_service
        self.stream_service = stream_service
        self.traffic_config_service = traffic_config_service
        self.traffic_counter = traffic_counter

        # ====================================================
        # PARÁMETROS
        # ====================================================

        self.conf = float(
            os.environ.get(
                "YOLO_CONF",
                "0.35",
            )
        )

        self.imgsz = int(
            os.environ.get(
                "YOLO_IMGSZ",
                "640",
            )
        )

        self.jpeg_quality = int(
            os.environ.get(
                "STREAM_JPEG_QUALITY",
                "75",
            )
        )

        self.target_fps = float(
            os.environ.get(
                "STREAM_TARGET_FPS",
                "15",
            )
        )

        self.tracker = os.environ.get(
            "YOLO_TRACKER",
            "bytetrack.yaml",
        )

        # ====================================================
        # ESTADO
        # ====================================================

        self.lock = threading.RLock()

        self.condition = threading.Condition(
            self.lock
        )

        self.running = False
        self.thread = None

        self.latest_jpeg = None

        # Este frame es el que consumirá WebRTC.
        self.latest_frame = None

        self.frame_sequence = 0

        self.frames = 0
        self.fps = 0.0

        self.error = None

        self.started_at = None

        self.frame_times = deque(
            maxlen=30
        )

    # ========================================================
    # INICIAR
    # ========================================================

    def start(self):
        with self.lock:
            if self.running:
                return

            if not self.stream_service.get_info()[
                "opened"
            ]:
                raise RuntimeError(
                    "No existe una fuente "
                    "de video abierta."
                )

            if not self.model_service.loaded:
                raise RuntimeError(
                    "El modelo YOLO no está cargado."
                )

            self.running = True

            self.frames = 0
            self.fps = 0.0
            self.error = None

            self.latest_jpeg = None
            self.latest_frame = None

            self.frame_sequence = 0

            self.frame_times.clear()

            self.started_at = time.monotonic()

            self.thread = threading.Thread(
                target=self._worker,
                name="vehicle-video-worker",
                daemon=True,
            )

            self.thread.start()

    # ========================================================
    # DETENER
    # ========================================================

    def stop(self):
        with self.condition:
            self.running = False
            self.condition.notify_all()

        thread = self.thread

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(
                timeout=2.0
            )

        with self.lock:
            self.thread = None

    # ========================================================
    # SALTO DE FRAMES
    # ========================================================

    def _frames_per_inference(self):
        source_fps = (
            self.stream_service.fps
        )

        if self.stream_service.is_file:
            return 1

        if (
            source_fps is None
            or source_fps <= 0
            or self.target_fps <= 0
        ):
            return 1

        ratio = (
            source_fps
            / self.target_fps
        )

        return max(
            1,
            int(round(ratio)),
        )

    # ========================================================
    # FPS
    # ========================================================

    def _update_fps(self):
        now = time.monotonic()

        self.frame_times.append(
            now
        )

        if len(self.frame_times) < 2:
            self.fps = 0.0
            return

        elapsed = (
            self.frame_times[-1]
            - self.frame_times[0]
        )

        if elapsed <= 0:
            return

        self.fps = (
            len(self.frame_times) - 1
        ) / elapsed

    # ========================================================
    # GEOMETRÍA
    # ========================================================

    def _draw_geometry(
        self,
        frame,
    ):
        height, width = (
            frame.shape[:2]
        )

        roi, line = (
            self.traffic_config_service.geometry(
                width,
                height,
            )
        )

        cv2.polylines(
            frame,
            [
                roi.reshape(
                    (-1, 1, 2)
                )
            ],
            True,
            (0, 255, 255),
            3,
        )

        cv2.putText(
            frame,
            "ROI",
            (
                int(roi[0][0]),
                max(
                    25,
                    int(roi[0][1]) - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        line_start = tuple(
            int(value)
            for value in line[0]
        )

        line_end = tuple(
            int(value)
            for value in line[1]
        )

        cv2.line(
            frame,
            line_start,
            line_end,
            (255, 0, 255),
            4,
        )

        label_x = int(
            (
                line_start[0]
                + line_end[0]
            )
            / 2
        )

        label_y = int(
            (
                line_start[1]
                + line_end[1]
            )
            / 2
        )

        cv2.putText(
            frame,
            "LINEA VIRTUAL",
            (
                label_x,
                max(
                    25,
                    label_y - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    def _draw_performance(
        self,
        frame,
    ):
        text = (
            f"GPU | FPS: "
            f"{self.fps:.1f}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX

        scale = 0.65
        thickness = 2

        (
            text_width,
            _,
        ), _ = cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )

        x = max(
            10,
            frame.shape[1]
            - text_width
            - 20,
        )

        cv2.putText(
            frame,
            text,
            (x, 28),
            font,
            scale,
            (0, 255, 0),
            thickness,
            cv2.LINE_AA,
        )

    # ========================================================
    # WORKER
    # ========================================================

    def _worker(self):
        frame_interval = (
            1.0 / self.target_fps
            if self.target_fps > 0
            else 0.0
        )

        frames_per_inference = (
            self._frames_per_inference()
        )

        print("=" * 65)
        print("VIDEO OPTIMIZADO / WEBRTC")
        print("=" * 65)

        print(
            "imgsz:",
            self.imgsz,
        )

        print(
            "conf:",
            self.conf,
        )

        print(
            "JPEG:",
            self.jpeg_quality,
        )

        print(
            "FPS objetivo:",
            self.target_fps,
        )

        print(
            "FPS fuente:",
            self.stream_service.fps,
        )

        print(
            "Frames fuente por inferencia:",
            frames_per_inference,
        )

        while True:
            with self.lock:
                if not self.running:
                    break

            iteration_start = (
                time.monotonic()
            )

            try:
                # ============================================
                # DESCARTAR FRAMES
                # ============================================

                if (
                    not self.stream_service.is_file
                    and frames_per_inference > 1
                ):
                    for _ in range(
                        frames_per_inference - 1
                    ):
                        self.stream_service.grab()

                # ============================================
                # LEER
                # ============================================

                frame = (
                    self.stream_service.read()
                )

                # ============================================
                # YOLO + BYTETRACK
                # ============================================

                result = (
                    self.model_service.model.track(
                        source=frame,
                        persist=True,
                        tracker=self.tracker,
                        conf=self.conf,
                        imgsz=self.imgsz,
                        device=self.model_service.device,
                        verbose=False,
                    )[0]
                )

                # ============================================
                # AFORO
                # ============================================

                self.traffic_counter.process(
                    result,
                    frame.shape,
                )

                # ============================================
                # ANOTADO
                # ============================================

                annotated = result.plot()

                self._draw_geometry(
                    annotated
                )

                with self.lock:
                    self.frames += 1
                    self._update_fps()

                self.traffic_counter.draw_overlay(
                    annotated
                )

                self._draw_performance(
                    annotated
                )

                # ============================================
                # GUARDAR FRAME BGR PARA WEBRTC
                # ============================================

                frame_webrtc = (
                    annotated.copy()
                )

                # ============================================
                # JPEG FALLBACK
                # ============================================

                ok, buffer = cv2.imencode(
                    ".jpg",
                    annotated,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        self.jpeg_quality,
                    ],
                )

                jpeg = (
                    buffer.tobytes()
                    if ok
                    else None
                )

                # ============================================
                # PUBLICAR
                # ============================================

                with self.condition:
                    self.latest_frame = (
                        frame_webrtc
                    )

                    if jpeg is not None:
                        self.latest_jpeg = jpeg

                    self.frame_sequence += 1

                    self.condition.notify_all()

            except Exception as exc:
                with self.condition:
                    self.error = str(exc)
                    self.running = False
                    self.condition.notify_all()

                print(
                    "ERROR VIDEO:",
                    exc,
                )

                break

            # ================================================
            # TARGET FPS
            # ================================================

            if frame_interval > 0:
                elapsed = (
                    time.monotonic()
                    - iteration_start
                )

                remaining = (
                    frame_interval
                    - elapsed
                )

                if remaining > 0:
                    time.sleep(
                        remaining
                    )

    # ========================================================
    # FRAME PARA WEBRTC
    # ========================================================

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None

            return (
                self.latest_frame.copy()
            )

    # ========================================================
    # MJPEG FALLBACK
    # ========================================================

    def generate(self):
        last_sequence = -1

        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: (
                        self.frame_sequence
                        != last_sequence
                        or not self.running
                    ),
                    timeout=2.0,
                )

                if (
                    not self.running
                    and self.latest_jpeg is None
                ):
                    break

                if (
                    self.frame_sequence
                    == last_sequence
                ):
                    if not self.running:
                        break

                    continue

                jpeg = self.latest_jpeg

                last_sequence = (
                    self.frame_sequence
                )

            if jpeg is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n"
                b"Pragma: no-cache\r\n"
                b"\r\n"
                + jpeg
                + b"\r\n"
            )

            if not self.running:
                break

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self):
        with self.lock:
            return {
                "running": self.running,
                "frames": self.frames,

                "fps": round(
                    self.fps,
                    2,
                ),

                "target_fps": self.target_fps,
                "imgsz": self.imgsz,
                "jpeg_quality": self.jpeg_quality,
                "conf": self.conf,

                "webrtc_frame_ready": (
                    self.latest_frame
                    is not None
                ),

                "error": self.error,
            }