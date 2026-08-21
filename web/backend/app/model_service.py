from pathlib import Path

import torch
from ultralytics import YOLO


class ModelService:
    def __init__(self, project_root: Path):
        self.project_root = project_root

        self.model_path = (
            project_root
            / "runs"
            / "entrenamiento"
            / "yolo11n_vehiculos_v2_dominio_objetivo_seed42-2"
            / "weights"
            / "best.pt"
        )

        self.model = None
        self.device = None
        self.loaded = False


    def load(self):
        if self.loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en: {self.model_path}"
            )

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

        self.loaded = True


    def get_info(self):
        return {
            "loaded": self.loaded,
            "model_path": str(self.model_path),
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "classes": (
                self.model.names
                if self.model is not None
                else {}
            ),
        }