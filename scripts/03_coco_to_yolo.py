# scripts/03_coco_to_yolo.py
import json
from pathlib import Path
from tqdm import tqdm
from configs.paths import DATASET_DIR, ANN_DIR

TRAIN_JSON = ANN_DIR / "instances_train.json"
VAL_JSON   = ANN_DIR / "instances_val.json"
CLASS_MAP  = ANN_DIR / "class_map.json"
LABELS_DIR = DATASET_DIR / "labels"

def load_coco(coco_path: Path):
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    img_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_img = {}
    for an in coco["annotations"]:
        anns_by_img.setdefault(an["image_id"], []).append(an)
    return img_by_id, anns_by_img

def load_class_map(class_map_path: Path):
    m = json.loads(class_map_path.read_text(encoding="utf-8"))
    return {int(k): int(v) for k, v in m["id_to_index"].items()}

def coco_box_to_yolo_line(an, W, H, id_to_index):
    x, y, w, h = an["bbox"]
    x = max(0.0, min(x, W - 1.0))
    y = max(0.0, min(y, H - 1.0))
    w = max(1.0, min(w, W - x))
    h = max(1.0, min(h, H - y))
    cx = (x + w / 2.0) / float(W)
    cy = (y + h / 2.0) / float(H)
    ww = w / float(W)
    hh = h / float(H)
    cx = max(0.0, min(cx, 1.0))
    cy = max(0.0, min(cy, 1.0))
    ww = max(1e-6, min(ww, 1.0))
    hh = max(1e-6, min(hh, 1.0))
    cls = id_to_index[int(an["category_id"])]
    return f"{cls} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}"

def save_split_yolo_labels(split_name: str, coco_json_path: Path, id_to_index: dict):
    labels_out = LABELS_DIR / split_name
    labels_out.mkdir(parents=True, exist_ok=True)
    img_by_id, anns_by_img = load_coco(coco_json_path)
    written = 0
    skipped = 0
    for img_id, im in tqdm(img_by_id.items(), desc=f"COCO→YOLO {split_name}"):
        W, H = int(im["width"]), int(im["height"])
        stem = Path(im["file_name"]).stem
        lines = []
        for an in anns_by_img.get(img_id, []):
            line = coco_box_to_yolo_line(an, W, H, id_to_index)
            if line is None:
                skipped += 1
                continue
            lines.append(line)
        (labels_out / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        written += 1
    print(f"[{split_name}] files:", written, "skipped_boxes:", skipped)

def main():
    assert TRAIN_JSON.exists(), f"missing {TRAIN_JSON}"
    assert VAL_JSON.exists(),   f"missing {VAL_JSON}"
    assert CLASS_MAP.exists(),  f"missing {CLASS_MAP}"
    id_to_index = load_class_map(CLASS_MAP)
    save_split_yolo_labels("train", TRAIN_JSON, id_to_index)
    save_split_yolo_labels("val",   VAL_JSON,   id_to_index)

if __name__ == "__main__":
    main()