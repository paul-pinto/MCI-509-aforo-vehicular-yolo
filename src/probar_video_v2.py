from pathlib import Path
from ultralytics import YOLO

ROOT = Path.cwd()

MODEL = (
    ROOT
    / "runs"
    / "entrenamiento"
    / "yolo11n_vehiculos_v2_dominio_objetivo_seed42-2"
    / "weights"
    / "best.pt"
)

# CAMBIA ESTO por la ruta real del video
VIDEO = ROOT / "videos" / "vtest.avi"

OUTPUT_PROJECT = ROOT / "resultados" / "video_v2"

if not MODEL.exists():
    raise FileNotFoundError(f"No existe el modelo:\n{MODEL}")

if not VIDEO.exists():
    raise FileNotFoundError(f"No existe el video:\n{VIDEO}")

print("=" * 72)
print("PRUEBA DE VIDEO COMPLETO - MODELO V2")
print("=" * 72)

print(f"Modelo : {MODEL}")
print(f"Video  : {VIDEO}")

model = YOLO(str(MODEL))

results = model.predict(
    source=str(VIDEO),

    imgsz=640,
    conf=0.25,
    iou=0.7,

    device=0,

    save=True,
    save_txt=False,
    save_conf=False,

    show_labels=True,
    show_conf=True,
    show_boxes=True,

    project=str(OUTPUT_PROJECT),
    name="deteccion_v2",

    exist_ok=True,

    stream=True,
    verbose=False,
)

frames = 0
detections = 0

for result in results:
    frames += 1

    if result.boxes is not None:
        detections += len(result.boxes)

    if frames % 100 == 0:
        print(
            f"Frames procesados: {frames} | "
            f"Detecciones acumuladas: {detections}"
        )

print()
print("=" * 72)
print("PROCESAMIENTO TERMINADO")
print("=" * 72)

print(f"Frames procesados     : {frames}")
print(f"Detecciones acumuladas: {detections}")

print(
    "\nResultado guardado en:\n"
    f"{OUTPUT_PROJECT / 'deteccion_v2'}"
)