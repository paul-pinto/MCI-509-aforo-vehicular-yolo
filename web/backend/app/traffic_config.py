import json
from pathlib import Path

import numpy as np


class TrafficConfigService:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_path = project_root / "config_trafico.json"

        self.config = None
        self.roi_raw = None
        self.line_raw = None

        self.load()

    def load(self):
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"No existe el archivo de configuración: {self.config_path}"
            )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as f:
            self.config = json.load(f)

        self.roi_raw = self._find_points(
            self.config,
            keywords=[
                "roi",
                "region",
                "poligono",
                "polygon",
            ],
        )

        self.line_raw = self._find_points(
            self.config,
            keywords=[
                "linea",
                "línea",
                "line",
                "conteo",
                "count",
            ],
        )

        if self.roi_raw is None:
            raise RuntimeError(
                "No se pudo localizar el ROI dentro de config_trafico.json."
            )

        if self.line_raw is None:
            raise RuntimeError(
                "No se pudo localizar la línea virtual "
                "dentro de config_trafico.json."
            )

        # Solo necesitamos dos puntos para la línea virtual
        self.line_raw = self.line_raw[:2]

    def _is_point_list(self, value):
        return (
            isinstance(value, list)
            and len(value) >= 2
            and all(
                isinstance(point, (list, tuple))
                and len(point) >= 2
                and isinstance(point[0], (int, float))
                and isinstance(point[1], (int, float))
                for point in value
            )
        )

    def _find_points(
        self,
        obj,
        keywords,
    ):
        if isinstance(obj, dict):
            # Primero priorizamos coincidencias por nombre de clave
            for key, value in obj.items():
                key_lower = str(key).lower()

                if (
                    any(
                        keyword in key_lower
                        for keyword in keywords
                    )
                    and self._is_point_list(value)
                ):
                    return value

            # Después buscamos de forma recursiva
            for value in obj.values():
                result = self._find_points(
                    value,
                    keywords,
                )

                if result is not None:
                    return result

        elif isinstance(obj, list):
            for value in obj:
                result = self._find_points(
                    value,
                    keywords,
                )

                if result is not None:
                    return result

        return None

    def scale_points(
        self,
        points,
        width,
        height,
    ):
        arr = np.array(
            points,
            dtype=np.float32,
        )

        # Si las coordenadas son normalizadas 0–1
        if np.nanmax(np.abs(arr)) <= 2.0:
            arr[:, 0] *= width
            arr[:, 1] *= height

        return np.rint(arr).astype(np.int32)

    def geometry(
        self,
        width,
        height,
    ):
        roi = self.scale_points(
            self.roi_raw,
            width,
            height,
        )

        line = self.scale_points(
            self.line_raw,
            width,
            height,
        )

        return roi, line

    def get_info(self):
        return {
            "config_path": str(self.config_path),
            "roi": self.roi_raw,
            "line": self.line_raw,
            "raw_config": self.config,
        }