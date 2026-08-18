from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path("datos/dominio_objetivo/roi_flujo_01_corregido")

IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"
CLASSES_FILE = BASE_DIR / "classes.txt"

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# CARGAR CLASES
# ============================================================

if not CLASSES_FILE.exists():
    raise FileNotFoundError(f"No existe: {CLASSES_FILE}")

classes = [
    line.strip()
    for line in CLASSES_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

print("=" * 70)
print("VALIDACIÓN DEL DATASET DEL DOMINIO OBJETIVO")
print("=" * 70)

print(f"\nClases encontradas: {len(classes)}")

for idx, class_name in enumerate(classes):
    print(f"  {idx}: {class_name}")


# ============================================================
# ARCHIVOS
# ============================================================

images = sorted(
    p for p in IMAGES_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
)

labels = sorted(
    p for p in LABELS_DIR.glob("*.txt")
    if p.name.lower() != "classes.txt"
)

image_stems = {p.stem for p in images}
label_stems = {p.stem for p in labels}


print("\n" + "-" * 70)
print("CONTEO")
print("-" * 70)

print(f"Imágenes : {len(images)}")
print(f"Labels   : {len(labels)}")


# ============================================================
# CORRESPONDENCIA IMAGE <-> LABEL
# ============================================================

missing_labels = sorted(image_stems - label_stems)
orphan_labels = sorted(label_stems - image_stems)

print("\n" + "-" * 70)
print("CORRESPONDENCIA")
print("-" * 70)

print(f"Imágenes sin label : {len(missing_labels)}")
print(f"Labels sin imagen  : {len(orphan_labels)}")

if missing_labels:
    print("\nImágenes sin label:")
    for stem in missing_labels:
        print(f"  - {stem}")

if orphan_labels:
    print("\nLabels sin imagen:")
    for stem in orphan_labels:
        print(f"  - {stem}")


# ============================================================
# VALIDACIÓN DEL CONTENIDO YOLO
# ============================================================

empty_labels = []
invalid_lines = []
invalid_classes = []
invalid_coordinates = []

total_objects = 0
objects_per_class = {i: 0 for i in range(len(classes))}


for label_path in labels:

    content = label_path.read_text(
        encoding="utf-8",
        errors="replace"
    ).strip()

    if not content:
        empty_labels.append(label_path.name)
        continue

    lines = content.splitlines()

    for line_number, line in enumerate(lines, start=1):

        parts = line.split()

        # YOLO detection:
        # class x_center y_center width height
        if len(parts) != 5:
            invalid_lines.append(
                (label_path.name, line_number, line)
            )
            continue

        try:
            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:
            invalid_lines.append(
                (label_path.name, line_number, line)
            )
            continue

        # Validar clase
        if class_id < 0 or class_id >= len(classes):
            invalid_classes.append(
                (label_path.name, line_number, class_id)
            )
            continue

        # Validar coordenadas normalizadas
        values = [x_center, y_center, width, height]

        if not all(0 <= value <= 1 for value in values):
            invalid_coordinates.append(
                (
                    label_path.name,
                    line_number,
                    x_center,
                    y_center,
                    width,
                    height,
                )
            )
            continue

        # width y height no deberían ser cero
        if width <= 0 or height <= 0:
            invalid_coordinates.append(
                (
                    label_path.name,
                    line_number,
                    x_center,
                    y_center,
                    width,
                    height,
                )
            )
            continue

        total_objects += 1
        objects_per_class[class_id] += 1


# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "-" * 70)
print("VALIDACIÓN DE ETIQUETAS")
print("-" * 70)

print(f"Labels vacíos          : {len(empty_labels)}")
print(f"Líneas inválidas       : {len(invalid_lines)}")
print(f"Clases inválidas       : {len(invalid_classes)}")
print(f"Coordenadas inválidas  : {len(invalid_coordinates)}")
print(f"Objetos válidos        : {total_objects}")


if empty_labels:
    print("\nLabels vacíos:")
    for item in empty_labels:
        print(f"  - {item}")


if invalid_lines:
    print("\nLíneas con formato inválido:")
    for filename, line_number, line in invalid_lines:
        print(
            f"  {filename}:{line_number}"
            f" -> {line}"
        )


if invalid_classes:
    print("\nClass IDs inválidos:")
    for filename, line_number, class_id in invalid_classes:
        print(
            f"  {filename}:{line_number}"
            f" -> class_id={class_id}"
        )


if invalid_coordinates:
    print("\nCoordenadas inválidas:")
    for (
        filename,
        line_number,
        x,
        y,
        w,
        h,
    ) in invalid_coordinates:

        print(
            f"  {filename}:{line_number}"
            f" -> x={x}, y={y}, w={w}, h={h}"
        )


# ============================================================
# DISTRIBUCIÓN POR CLASE
# ============================================================

print("\n" + "-" * 70)
print("DISTRIBUCIÓN DE OBJETOS")
print("-" * 70)

for class_id, count in objects_per_class.items():
    print(
        f"{class_id:>2} | "
        f"{classes[class_id]:<15} | "
        f"{count:>5}"
    )


# ============================================================
# RESULTADO GENERAL
# ============================================================

has_errors = any([
    missing_labels,
    orphan_labels,
    empty_labels,
    invalid_lines,
    invalid_classes,
    invalid_coordinates,
])

print("\n" + "=" * 70)

if has_errors:
    print("RESULTADO: DATASET CON OBSERVACIONES")
else:
    print("RESULTADO: DATASET VÁLIDO")

print("=" * 70)