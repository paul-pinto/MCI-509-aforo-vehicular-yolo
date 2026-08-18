from pathlib import Path
from ultralytics import YOLO

ROOT = Path.cwd()

V1 = (
    ROOT
    / "runs"
    / "entrenamiento"
    / "yolo11n_vehiculos_definitivo_seed42"
    / "weights"
    / "best.pt"
)

V2 = (
    ROOT
    / "runs"
    / "entrenamiento"
    / "yolo11n_vehiculos_v2_dominio_objetivo_seed42-2"
    / "weights"
    / "best.pt"
)

SRI_YAML = (
    ROOT
    / "resultados"
    / "evaluacion"
    / "_config"
    / "data_test_resuelto.yaml"
)

TARGET_ROOT = (
    ROOT
    / "datos"
    / "dominio_objetivo"
    / "split_flujo_01"
)

TARGET_YAML = (
    ROOT
    / "datos"
    / "dominio_objetivo"
    / "data_target_eval.yaml"
)

# ------------------------------------------------------------
# Crear YAML del dominio objetivo
# ------------------------------------------------------------

yaml_content = f"""path: {TARGET_ROOT.as_posix()}

train: train/images
val: val/images
test: test/images

names:
  0: car
  1: threewheel
  2: bus
  3: truck
  4: motorbike
  5: van
"""

TARGET_YAML.write_text(
    yaml_content,
    encoding="utf-8"
)


def evaluar(nombre, model_path, data_yaml, split):
    print()
    print("=" * 80)
    print(nombre)
    print("=" * 80)

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        plots=True,
        verbose=True,
    )

    print()
    print(f"Precision    : {metrics.box.mp:.6f}")
    print(f"Recall       : {metrics.box.mr:.6f}")
    print(f"mAP50        : {metrics.box.map50:.6f}")
    print(f"mAP50-95     : {metrics.box.map:.6f}")

    return metrics


# ============================================================
# TEST SRI LANKA
# ============================================================

evaluar(
    "V1 - TEST SRI LANKA",
    V1,
    SRI_YAML,
    "test",
)

evaluar(
    "V2 - TEST SRI LANKA",
    V2,
    SRI_YAML,
    "test",
)


# ============================================================
# TEST EXTERNO DOMINIO OBJETIVO
# ============================================================

evaluar(
    "V1 - TEST DOMINIO OBJETIVO",
    V1,
    TARGET_YAML,
    "test",
)

evaluar(
    "V2 - TEST DOMINIO OBJETIVO",
    V2,
    TARGET_YAML,
    "test",
)