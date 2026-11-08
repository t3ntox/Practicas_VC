import json, random
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps


DATASET_DIR = Path("./raw-dataset")
IMG_DIR     = DATASET_DIR / "images"
ANN_DIR     = DATASET_DIR / "annotations"

OUT_DIR     = Path("./coco-dataset")
Path.mkdir(OUT_DIR, exist_ok=True)
OUT_IMG_DIR = OUT_DIR / "images"
Path.mkdir(OUT_IMG_DIR, exist_ok=True)
OUT_ANN_DIR = OUT_DIR / "annotations"
Path.mkdir(OUT_ANN_DIR, exist_ok=True)

MAX_SIDE = 1280                                        # lado mayor tras reescalar
BLOCK    = 10                                          # fotos seguidas que van juntas al mismo conjunto
SPLITS   = {"train": 0.60, "val": 0.15, "test": 0.25}
SEED     = 42

CATEGORIES = [{"id": 0, "name": "matricula", "supercategory": "vehiculo"}]
IMG_EXTS   = {".jpg", ".jpeg", ".png"}

import statistics

def rect_to_xyxy(rect, W, H):
    """[cx, cy, w, h] normalizado -> esquinas en píxeles, recortadas a la imagen."""
    cx, cy, w, h = rect
    x1 = max(0.0, (cx - w / 2) * W)
    y1 = max(0.0, (cy - h / 2) * H)
    x2 = min(float(W), (cx + w / 2) * W)
    y2 = min(float(H), (cy + h / 2) * H)
    return x1, y1, x2, y2


def load_resized(path):
    """Abre la imagen, aplica la rotación EXIF y reduce el lado mayor a MAX_SIDE (no amplía)."""
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    W, H = im.size
    scale = min(1.0, MAX_SIDE / max(W, H))
    if scale < 1.0:
        im = im.resize((round(W * scale), round(H * scale)), Image.LANCZOS)
    return im


OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_ANN_DIR.mkdir(parents=True, exist_ok=True)

coco_images, coco_anns = [], []
sin_anotacion, cajas_degeneradas = [], 0

img_paths = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
for img_path in img_paths:
    ann_path = ANN_DIR / f"{img_path.stem}.json"
    if not ann_path.exists():
        sin_anotacion.append(img_path.name)
        continue

    out_name = f"{img_path.stem}.jpg"
    out_path = OUT_IMG_DIR / out_name
    if out_path.exists():                       # ya procesada en una ejecución anterior
        W, H = Image.open(out_path).size
    else:
        im = load_resized(img_path)
        im.save(out_path, quality=95)           # se guarda sin EXIF: la orientación ya está aplicada
        W, H = im.size

    img_id = len(coco_images)
    coco_images.append({"id": img_id, "file_name": out_name, "width": W, "height": H,
                        "orig_name": img_path.name})

    objs = json.loads(ann_path.read_text(encoding="utf-8"))
    # Algunas fotos con EXIF de orientación 6 se anotaron sobre la imagen SIN girar: todas sus
    # cajas salen verticales (ancho/alto << 1). Se detectan por la mediana de la proporción y se
    # llevan a la imagen girada (90° horario): cx' = 1 - cy, cy' = cx, ancho y alto intercambiados.
    orient  = Image.open(img_path).getexif().get(0x0112, 1)
    ratios  = [o["rectangle"][2] * W / (o["rectangle"][3] * H) for o in objs if o["rectangle"][2] * o["rectangle"][3] > 0]
    anotacion_girada = orient == 6 and bool(ratios) and statistics.median(ratios) < 0.6

    for obj in objs:
        cx, cy, rw, rh = obj["rectangle"]
        rect = [1 - cy, cx, rh, rw] if anotacion_girada else obj["rectangle"]
        x1, y1, x2, y2 = rect_to_xyxy(rect, W, H)
        w, h = x2 - x1, y2 - y1
        if w < 1 or h < 1:
            cajas_degeneradas += 1
            continue
        plate   = obj.get("plate")
        bbox    = [round(v, 2) for v in (x1, y1, w, h)]
        legible = plate is not None and plate.strip().upper() not in ("", "NONE")   # "None" y "NONE" = ilegible
        text    = plate.strip().upper() if legible else None
        coco_anns.append({
            "id": len(coco_anns),
            "image_id": img_id,
            "category_id": 0,
            "bbox": bbox,
            "area": round(bbox[2] * bbox[3], 2),    # área coherente con la caja redondeada
            "iscrowd": 0,
            "segmentation": [[round(v, 2) for v in (x1, y1, x2, y1, x2, y2, x1, y2)]],
            "text": text,
            "chars": list(text) if legible else [],
            "legible": legible,
            "anotacion_girada": anotacion_girada,
        })


blocks = [coco_images[i:i + BLOCK] for i in range(0, len(coco_images), BLOCK)]
order = list(range(len(blocks)))
random.Random(SEED).shuffle(order)

n_total = len(coco_images)
limits  = {"train": SPLITS["train"] * n_total,
           "val":   (SPLITS["train"] + SPLITS["val"]) * n_total}

split_of_img, assigned = {}, 0
for b in order:
    split = "train" if assigned < limits["train"] else "val" if assigned < limits["val"] else "test"
    for img in blocks[b]:
        split_of_img[img["id"]] = split
    assigned += len(blocks[b])

coco_splits = {}
for split in SPLITS:
    imgs = [i for i in coco_images if split_of_img[i["id"]] == split]
    ids  = {i["id"] for i in imgs}
    anns = [a for a in coco_anns if a["image_id"] in ids]
    coco_splits[split] = {
        "info": {"description": f"Matrículas - {split}", "max_side": MAX_SIDE, "seed": SEED, "block": BLOCK},
        "images": imgs,
        "annotations": anns,
        "categories": CATEGORIES,
    }
    (OUT_ANN_DIR / f"instances_{split}.json").write_text(json.dumps(coco_splits[split], ensure_ascii=False),
                                                         encoding="utf-8")

# Carpeta de imágenes propia para cada conjunto (copias de OUT_IMG_DIR)
import shutil
for split, d in coco_splits.items():
    split_dir = OUT_DIR / f"images_{split}"
    split_dir.mkdir(exist_ok=True)
    for img in d["images"]:
        dst = split_dir / img["file_name"]
        if not dst.exists():
            shutil.copy2(OUT_IMG_DIR / img["file_name"], dst)

# Comprobación: cada bloque cae entero en un único conjunto
assert all(len({split_of_img[i["id"]] for i in blk}) == 1 for blk in blocks)

print(f"{'conjunto':<6} {'imágenes':>9} {'%':>6} {'matrículas':>11} {'legibles':>9}")
for split, d in coco_splits.items():
    n_img = len(d["images"])
    print(f"{split:<6} {n_img:>9} {100 * n_img / n_total:>5.1f}% {len(d['annotations']):>11} "
          f"{sum(a['legible'] for a in d['annotations']):>9}")
print(f"\nJSON guardados en {OUT_ANN_DIR}")

print(f"Imágenes convertidas: {len(coco_images)}")
print(f"Matrículas: {len(coco_anns)}  (legibles: {sum(a['legible'] for a in coco_anns)})")
print(f"Sin anotación (excluidas): {sin_anotacion}")
print(f"Cajas descartadas por tamaño < 1 px: {cajas_degeneradas}")
print("Tamaños más frecuentes:", Counter((i['width'], i['height']) for i in coco_images).most_common(5))