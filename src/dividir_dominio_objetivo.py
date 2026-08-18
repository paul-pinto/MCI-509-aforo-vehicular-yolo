from pathlib import Path
from collections import Counter
import shutil
import re

# ============================================================
# CONFIGURACIÓN
# ============================================================

SOURCE = Path("datos/dominio_objetivo/roi_flujo_01_corregido")
DEST = Path("datos/dominio_objetivo/split_flujo_01")

IMAGES_DIR = SOURCE / "images"
LABELS_DIR = SOURCE / "labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_N = 80
VAL_N = 10
TEST_N = 10


# ============================================================
# FUNCIÓN PARA EXTRAER ÍNDICE DEL FRAME
# ============================================================

def frame_index(path: Path) -> int:
    """
    Espera nombres como:
        flujo_01_0001_20260814_133909.jpg
        flujo_01_0042_20260814_....
    """

    match = re.search(r"flujo_01_(\d+)", path.stem)

    if not match:
        raise ValueError(
            f"No pude extraer el índice del frame de: {path.name}"
        )

    return int(match.group(1))


# ============================================================
# CARGAR IMÁGENES
# ============================================================

images = [
    p
    for p in IMAGES_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

images = sorted(images, key=frame_index)

if len(images) != TRAIN_N + VAL_N + TEST_N:
    raise RuntimeError(
        f"Se esperaban {TRAIN_N + VAL_N + TEST_N} imágenes, "
        f"pero se encontraron {len(images)}."
    )


# ============================================================
# CREAR SPLITS TEMPORALES
# ============================================================

splits = {
    "train": images[:TRAIN_N],
    "val": images[TRAIN_N:TRAIN_N + VAL_N],
    "test": images[TRAIN_N + VAL_N:],
}


# ============================================================
# LIMPIAR DESTINO ANTERIOR
# ============================================================

if DEST.exists():
    print(f"Eliminando split anterior: {DEST}")
    shutil.rmtree(DEST)


# ============================================================
# COPIAR IMÁGENES Y LABELS
# ============================================================

for split_name, split_images in splits.items():

    dest_images = DEST / split_name / "images"
    dest_labels = DEST / split_name / "labels"

    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    for image_path in split_images:

        label_path = LABELS_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"Falta label para {image_path.name}: {label_path}"
            )

        shutil.copy2(
            image_path,
            dest_images / image_path.name
        )

        shutil.copy2(
            label_path,
            dest_labels / label_path.name
        )


# ============================================================
# ANALIZAR DISTRIBUCIÓN
# ============================================================

def count_objects(split_name):
    counter = Counter()

    labels_dir = DEST / split_name / "labels"

    for label_file in labels_dir.glob("*.txt"):

        for line in label_file.read_text(
            encoding="utf-8"
        ).splitlines():

            line = line.strip()

            if not line:
                continue

            class_id = int(line.split()[0])
            counter[class_id] += 1

    return counter


CLASSES = {
    0: "car",
    1: "threewheel",
    2: "bus",
    3: "truck",
    4: "motorbike",
    5: "van",
}


# ============================================================
# REPORTE
# ============================================================

print("=" * 72)
print("SPLIT TEMPORAL DEL DOMINIO OBJETIVO")
print("=" * 72)

for split_name, split_images in splits.items():

    first = split_images[0]
    last = split_images[-1]

    counts = count_objects(split_name)
    total_objects = sum(counts.values())

    print()
    print("-" * 72)
    print(split_name.upper())
    print("-" * 72)

    print(f"Imágenes      : {len(split_images)}")
    print(f"Primer frame  : {first.name}")
    print(f"Último frame  : {last.name}")
    print(f"Objetos       : {total_objects}")

    print("\nDistribución:")

    for class_id, class_name in CLASSES.items():
        count = counts.get(class_id, 0)

        print(
            f"  {class_id} | "
            f"{class_name:<12} | "
            f"{count:>5}"
        )


# ============================================================
# COMPROBAR TOTALES
# ============================================================

total_copied_images = sum(
    len(list((DEST / split / "images").iterdir()))
    for split in splits
)

total_copied_labels = sum(
    len(list((DEST / split / "labels").glob("*.txt")))
    for split in splits
)

print()
print("=" * 72)
print("RESUMEN")
print("=" * 72)

print(f"Imágenes originales : {len(images)}")
print(f"Imágenes copiadas   : {total_copied_images}")
print(f"Labels copiados     : {total_copied_labels}")

if (
    total_copied_images == 100
    and total_copied_labels == 100
):
    print("\nRESULTADO: SPLIT CREADO CORRECTAMENTE")
else:
    print("\nRESULTADO: REVISAR SPLIT")