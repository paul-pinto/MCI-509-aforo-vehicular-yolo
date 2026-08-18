from pathlib import Path
from ultralytics import YOLO
import torch


# ============================================================
# RUTAS
# ============================================================

ROOT = Path.cwd()

MODEL_PATH = (
    ROOT
    / "runs"
    / "entrenamiento"
    / "yolo11n_vehiculos_definitivo_seed42"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    ROOT
    / "datos"
    / "dataset_v2"
    / "data.yaml"
)

PROJECT_DIR = (
    ROOT
    / "runs"
    / "entrenamiento"
)

RUN_NAME = "yolo11n_vehiculos_v2_dominio_objetivo_seed42"


# ============================================================
# VALIDACIONES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo anterior:\n{MODEL_PATH}"
    )

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"No se encontró data.yaml:\n{DATA_YAML}"
    )


# ============================================================
# INFORMACIÓN DEL ENTORNO
# ============================================================

print("=" * 72)
print("SEGUNDO FINE-TUNING - YOLO11N")
print("=" * 72)

print(f"Modelo inicial : {MODEL_PATH}")
print(f"Dataset V2     : {DATA_YAML}")
print(f"CUDA           : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")

print()


# ============================================================
# CARGAR MODELO PREVIAMENTE AJUSTADO
# ============================================================

model = YOLO(str(MODEL_PATH))


# ============================================================
# FINE-TUNING V2
# ============================================================

results = model.train(
    data=str(DATA_YAML),

    # --------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------
    epochs=30,
    patience=10,

    imgsz=640,
    batch=4,

    device=0,
    workers=0,

    # --------------------------------------------------------
    # Reproducibilidad
    # --------------------------------------------------------
    seed=42,
    deterministic=True,

    # --------------------------------------------------------
    # Optimizador
    # --------------------------------------------------------
    optimizer="AdamW",

    # LR menor porque partimos del best.pt ya especializado
    lr0=0.001,
    lrf=0.01,

    weight_decay=0.0005,

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    translate=0.1,
    scale=0.5,

    fliplr=0.5,

    mosaic=1.0,
    close_mosaic=10,

    # --------------------------------------------------------
    # Otros
    # --------------------------------------------------------
    amp=True,
    cache=False,

    val=True,
    plots=True,

    save=True,
    save_period=-1,

    project=str(PROJECT_DIR),
    name=RUN_NAME,

    exist_ok=False,

    verbose=True,
)


print()
print("=" * 72)
print("ENTRENAMIENTO TERMINADO")
print("=" * 72)

print(
    "Resultados guardados en:\n"
    f"{PROJECT_DIR / RUN_NAME}"
)