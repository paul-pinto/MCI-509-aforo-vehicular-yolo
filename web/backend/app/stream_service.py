from pathlib import Path

import cv2


class StreamService:
    def __init__(self):
        self.original_source = None
        self.source = None

        self.cap = None

        self.is_file = False

        self.fps = None
        self.width = None
        self.height = None

        self.resolver = None

        self.metadata = {}

    # ========================================================
    # DETECTAR ARCHIVO LOCAL
    # ========================================================

    @staticmethod
    def _is_local_file(
        source: str,
    ):
        if "://" in source:
            return False

        try:
            return Path(
                source
            ).exists()

        except Exception:
            return False

    # ========================================================
    # ABRIR
    # ========================================================

    def open(
        self,
        source: str,
        *,
        original_source: str = None,
        resolver: str = "direct",
        metadata: dict | None = None,
    ):
        self.close()

        self.original_source = (
            original_source
            or source
        )

        self.source = source

        self.resolver = resolver

        self.metadata = (
            dict(metadata)
            if metadata
            else {}
        )

        self.is_file = (
            self._is_local_file(
                source
            )
        )

        # ====================================================
        # FFMPEG
        # ====================================================

        self.cap = cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
        )

        if (
            self.cap is not None
        ):
            # OpenCV/FFmpeg no siempre respeta
            # CAP_PROP_BUFFERSIZE, pero cuando
            # lo hace ayuda mucho con streams
            # en vivo porque evita acumular
            # frames antiguos.
            try:
                self.cap.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1,
                )

            except Exception:
                pass

        # ====================================================
        # FALLBACK
        # ====================================================

        if (
            self.cap is None
            or not self.cap.isOpened()
        ):
            if self.cap is not None:
                self.cap.release()

            self.cap = (
                cv2.VideoCapture(
                    source
                )
            )

            if (
                self.cap is not None
            ):
                try:
                    self.cap.set(
                        cv2.CAP_PROP_BUFFERSIZE,
                        1,
                    )

                except Exception:
                    pass

        if (
            self.cap is None
            or not self.cap.isOpened()
        ):
            self.cap = None

            raise RuntimeError(
                "No se pudo abrir "
                "la fuente de video resuelta."
            )

        # ====================================================
        # INFORMACIÓN DE FUENTE
        # ====================================================

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        width = self.cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )

        height = self.cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )

        self.fps = (
            float(fps)
            if fps
            and fps > 0
            else None
        )

        self.width = (
            int(width)
            if width
            and width > 0
            else None
        )

        self.height = (
            int(height)
            if height
            and height > 0
            else None
        )

        # ====================================================
        # PRUEBA DE LECTURA
        # ====================================================

        ok, frame = (
            self.cap.read()
        )

        if (
            not ok
            or frame is None
        ):
            self.cap.release()

            self.cap = None

            raise RuntimeError(
                "La fuente pudo abrirse, "
                "pero OpenCV no pudo "
                "decodificar ningún frame."
            )

        self.height = int(
            frame.shape[0]
        )

        self.width = int(
            frame.shape[1]
        )

        # En archivos locales volvemos
        # al inicio.
        if self.is_file:
            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0,
            )

    # ========================================================
    # LEER FRAME
    # ========================================================

    def read(self):
        if self.cap is None:
            raise RuntimeError(
                "No hay una fuente "
                "de video abierta."
            )

        ok, frame = (
            self.cap.read()
        )

        # ====================================================
        # LOOP PARA ARCHIVOS
        # ====================================================

        if (
            (
                not ok
                or frame is None
            )
            and self.is_file
        ):
            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0,
            )

            ok, frame = (
                self.cap.read()
            )

        if (
            not ok
            or frame is None
        ):
            raise RuntimeError(
                "No se pudo leer "
                "un frame de la fuente."
            )

        return frame

    # ========================================================
    # DESCARTAR FRAME
    # ========================================================

    def grab(self):
        if self.cap is None:
            return False

        try:
            return bool(
                self.cap.grab()
            )

        except Exception:
            return False

    # ========================================================
    # CERRAR
    # ========================================================

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()

            except Exception:
                pass

        self.cap = None

        self.original_source = None
        self.source = None

        self.is_file = False

        self.fps = None
        self.width = None
        self.height = None

        self.resolver = None

        self.metadata = {}

    # ========================================================
    # ESTADO
    # ========================================================

    def get_info(self):
        info = {
            "opened": (
                self.cap is not None
                and self.cap.isOpened()
            ),

            "source": (
                self.original_source
            ),

            "resolver": (
                self.resolver
            ),

            "type": (
                "file"
                if self.is_file
                else "stream"
            ),

            "fps_source": (
                self.fps
            ),

            "width": (
                self.width
            ),

            "height": (
                self.height
            ),
        }

        info.update(
            self.metadata
        )

        return info