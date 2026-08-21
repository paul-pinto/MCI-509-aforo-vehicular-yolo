import time

import cv2


class VideoService:
    def __init__(
        self,
        model_service,
        stream_service,
        traffic_config_service,
        traffic_counter,
    ):
        self.model_service = (
            model_service
        )

        self.stream_service = (
            stream_service
        )

        self.traffic_config_service = (
            traffic_config_service
        )

        self.traffic_counter = (
            traffic_counter
        )

        self.running = False

        self.conf = 0.35
        self.imgsz = 768

        self.frames = 0
        self.start_time = None

    # ========================================================
    # CONTROL
    # ========================================================

    def start(self):
        if not (
            self.stream_service
            .get_info()["opened"]
        ):
            raise RuntimeError(
                "No hay una fuente "
                "de video abierta."
            )

        self.running = True
        self.frames = 0
        self.start_time = (
            time.time()
        )

    def stop(self):
        self.running = False

    # ========================================================
    # ESTADO
    # ========================================================

    def get_status(self):
        fps = 0.0

        if self.start_time is not None:
            elapsed = (
                time.time()
                - self.start_time
            )

            if elapsed > 0:
                fps = (
                    self.frames
                    / elapsed
                )

        return {
            "running": self.running,
            "frames": self.frames,
            "fps": round(
                fps,
                2,
            ),
        }

    # ========================================================
    # ROI + LÍNEA
    # ========================================================

    def draw_traffic_geometry(
        self,
        frame,
    ):
        height, width = (
            frame.shape[:2]
        )

        roi, line = (
            self.traffic_config_service
            .geometry(
                width,
                height,
            )
        )

        # ROI amarillo
        cv2.polylines(
            frame,
            [
                roi.reshape(
                    (-1, 1, 2)
                )
            ],
            isClosed=True,
            color=(0, 255, 255),
            thickness=3,
        )

        cv2.putText(
            frame,
            "ROI",
            tuple(roi[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Línea magenta
        cv2.line(
            frame,
            tuple(line[0]),
            tuple(line[1]),
            (255, 0, 255),
            4,
        )

        cv2.putText(
            frame,
            "LINEA VIRTUAL",
            tuple(line[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # ========================================================
    # GENERADOR MJPEG
    # ========================================================

    def generate(self):
        if not self.running:
            self.start()

        while self.running:
            try:
                frame = (
                    self.stream_service
                    .read()
                )

            except Exception:
                time.sleep(0.05)
                continue

            # =================================================
            # YOLO11 + BYTETRACK
            # =================================================

            result = (
                self.model_service
                .model
                .track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=(
                        self.model_service
                        .device
                    ),
                    verbose=False,
                )[0]
            )

            # =================================================
            # LÓGICA DE AFORO
            # =================================================

            self.traffic_counter.process(
                result=result,
                frame_shape=frame.shape,
            )

            # =================================================
            # DIBUJAR DETECCIONES
            # =================================================

            annotated = (
                result.plot()
            )

            # =================================================
            # ROI + LÍNEA
            # =================================================

            self.draw_traffic_geometry(
                annotated
            )

            # =================================================
            # PANEL DE AFORO
            # =================================================

            self.traffic_counter.draw_overlay(
                annotated
            )

            # =================================================
            # FPS
            # =================================================

            self.frames += 1

            elapsed = (
                time.time()
                - self.start_time
            )

            fps = (
                self.frames / elapsed
                if elapsed > 0
                else 0.0
            )

            cv2.putText(
                annotated,
                f"GPU | FPS: {fps:.1f}",
                (
                    annotated.shape[1]
                    - 220,
                    32,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # =================================================
            # JPEG
            # =================================================

            ok, buffer = (
                cv2.imencode(
                    ".jpg",
                    annotated,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        82,
                    ],
                )
            )

            if not ok:
                continue

            # =================================================
            # MJPEG
            # =================================================

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )