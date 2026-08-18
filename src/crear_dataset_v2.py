from pathlib import Path
import shutil
from collections import Counter

# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path.cwd()

SRI_LANKA = ROOT / "vehicle dataset limpio final"
TARGET = ROOT / "datos" / "dominio_objetivo" / "split_flujo_01"

DEST = ROOT / "datos" / "dataset_v2"

CLASSES = {
    0: "car",
    1: "threewheel",
    2: "bus",
    3: "truck",
    4: "motorbike",
    5: "van",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# UTILIDADES
# ============================================================

def get_images(directory: Path):
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_pair(image_path: Path, label_path: Path,
              dest_images: Path, dest_labels: Path,
              prefix: str = ""):

    if not label_path.exists():
        raise FileNotFoundError(
            f"Falta label para {image_path.name}: {label_path}"
        )

    image_name = f"{prefix}{image_path.name}"
    label_name = f"{prefix}{label_path.name}"

    shutil.copy2(
        image_path,
        dest_images / image_name
    )

    shutil.copy2(
        label_path,
        dest_labels / label_name
    )


def count_objects(labels_dir: Path):
    counter = Counter()

    for label_path in labels_dir.glob("*.txt"):

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():

            line = line.strip()

            if not line:
                continue

            class_id = int(line.split()[0])
            counter[class_id] += 1

    return counter


# ============================================================
# VALIDAR FUENTES
# ============================================================

required_paths = [
    SRI_LANKA / "images" / "train",
    SRI_LANKA / "images" / "val",
    SRI_LANKA / "images" / "test",
    SRI_LANKA / "labels" / "train",
    SRI_LANKA / "labels" / "val",
    SRI_LANKA / "labels" / "test",

    TARGET / "train" / "images",
    TARGET / "train" / "labels",
    TARGET / "val" / "images",
    TARGET / "val" / "labels",
    TARGET / "test" / "images",
    TARGET / "test" / "labels",
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe la ruta requerida:\n{path}"
        )


# ============================================================
# LIMPIAR DESTINO
# ============================================================

if DEST.exists():
    print(f"Eliminando dataset V2 anterior: {DEST}")
    shutil.rmtree(DEST)


# ============================================================
# CREAR DIRECTORIOS
# ============================================================

for split in ("train", "val", "test"):

    (DEST / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (DEST / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 1. COPIAR SRI LANKA
# ============================================================

print("=" * 72)
print("COPIANDO DATASET SRI LANKA")
print("=" * 72)

for split in ("train", "val", "test"):

    source_images = SRI_LANKA / "images" / split
    source_labels = SRI_LANKA / "labels" / split

    dest_images = DEST / "images" / split
    dest_labels = DEST / "labels" / split

    images = get_images(source_images)

    for image_path in images:

        label_path = source_labels / f"{image_path.stem}.txt"

        copy_pair(
            image_path,
            label_path,
            dest_images,
            dest_labels,
            prefix="sl_"
        )

    print(
        f"{split:<5}: {len(images)} imágenes Sri Lanka"
    )


# ============================================================
# 2. AÑADIR DOMINIO OBJETIVO A TRAIN Y VAL
# ============================================================

print()
print("=" * 72)
print("AÑADIENDO DOMINIO OBJETIVO")
print("=" * 72)

for split in ("train", "val"):

    source_images = TARGET / split / "images"
    source_labels = TARGET / split / "labels"

    dest_images = DEST / "images" / split
    dest_labels = DEST / "labels" / split

    images = get_images(source_images)

    for image_path in images:

        label_path = source_labels / f"{image_path.stem}.txt"

        copy_pair(
            image_path,
            label_path,
            dest_images,
            dest_labels,
            prefix="target_"
        )

    print(
        f"{split:<5}: +{len(images)} imágenes dominio objetivo"
    )


# ============================================================
# 3. CREAR DATA.YAML
# ============================================================

yaml_content = """path: .
train: images/train
val: images/val
test: images/test

names:
  0: car
  1: threewheel
  2: bus
  3: truck
  4: motorbike
  5: van
"""

(DEST / "data.yaml").write_text(
    yaml_content,
    encoding="utf-8"
)


# ============================================================
# 4. REPORTE FINAL
# ============================================================

print()
print("=" * 72)
print("DATASET V2")
print("=" * 72)

for split in ("train", "val", "test"):

    images_dir = DEST / "images" / split
    labels_dir = DEST / "labels" / split

    images = get_images(images_dir)
    labels = list(labels_dir.glob("*.txt"))

    counts = count_objects(labels_dir)
    total_objects = sum(counts.values())

    print()
    print("-" * 72)
    print(split.upper())
    print("-" * 72)

    print(f"Imágenes : {len(images)}")
    print(f"Labels   : {len(labels)}")
    print(f"Objetos  : {total_objects}")

    print("\nDistribución:")

    for class_id, class_name in CLASSES.items():

        print(
            f"  {class_id} | "
            f"{class_name:<12} | "
            f"{counts.get(class_id, 0):>5}"
        )


# ============================================================
# 5. TEST EXTERNO
# ============================================================

external_test_images = get_images(
    TARGET / "test" / "images"
)

external_test_labels = list(
    (TARGET / "test" / "labels").glob("*.txt")
)

print()
print("=" * 72)
print("TEST EXTERNO DEL DOMINIO OBJETIVO")
print("=" * 72)

print(
    f"Imágenes : {len(external_test_images)}"
)

print(
    f"Labels   : {len(external_test_labels)}"
)

print()
print(f"Dataset V2 creado en:")
print(DEST)

print()
print("RESULTADO: DATASET V2 CREADO CORRECTAMENTE")