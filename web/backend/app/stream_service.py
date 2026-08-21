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

        source_path = Path(
            source
        )

        self.is_file = (
            source_path.exists()
        )

        # ====================================================
        # INTENTO CON FFMPEG
        # ====================================================

        self.cap = cv2.VideoCapture(
            source,
            cv2.CAP_FFMPEG,
        )

        # ====================================================
        # FALLBACK AUTOMÁTICO
        # ====================================================

        if not self.cap.isOpened():
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(
                source
            )

        if not self.cap.isOpened():
            self.cap = None

            raise RuntimeError(
                "No se pudo abrir "
                "la fuente de video resuelta."
            )

        # ====================================================
        # INFORMACIÓN DE LA FUENTE
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
        # PROBAR LECTURA REAL
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

        # Usamos las dimensiones reales
        # del primer frame.
        self.height = int(
            frame.shape[0]
        )

        self.width = int(
            frame.shape[1]
        )

        # Para archivos locales,
        # volver al comienzo.
        if self.is_file:
            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0,
            )

    # ========================================================
    # LEER
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
        # LOOP PARA ARCHIVOS LOCALES
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
    # CERRAR
    # ========================================================

    def close(self):
        if self.cap is not None:
            self.cap.release()

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
    # INFO
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