# scripts/01_merge_to_coco.py
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from configs.paths import RAW_TRAIN_IMG, RAW_ANN_DIR

# 출력 경로
MERGED_COCO_JSON = RAW_ANN_DIR / "train_annotations_merged.json"

IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

def norm_bbox(b):
    # 다양한 키에서 [x,y,w,h] 표준화
    if isinstance(b, (list, tuple)) and len(b) == 4:
        x, y, w, h = b
        return float(x), float(y), float(w), float(h)
    if isinstance(b, dict):
        x = b.get("x", b.get("xmin", b.get("left")))
        y = b.get("y", b.get("ymin", b.get("top")))
        w = b.get("w", b.get("width"))
        h = b.get("h", b.get("height"))
        if None not in (x, y, w, h):
            return float(x), float(y), float(w), float(h)
    return None

def resolve_image_path(fname: str | None, jf: Path) -> Path | None:
    # 파일명 해석 우선순위: fname → stem 동일 → 하이픈/언더스코어 변형
    if fname:
        p = RAW_TRAIN_IMG / Path(fname).name
        if p.exists():
            return p
    stem = (Path(fname).stem if fname else jf.stem)
    for ext in IMG_EXTS:
        p = RAW_TRAIN_IMG / f"{stem}{ext}"
        if p.exists():
            return p
    alt = stem.replace("-", "_")
    for ext in IMG_EXTS:
        p = RAW_TRAIN_IMG / f"{alt}{ext}"
        if p.exists():
            return p
    return None

def main():
    json_files = list(RAW_ANN_DIR.rglob("*.json"))
    assert json_files, f"annotation json not found under {RAW_ANN_DIR}"

    images, annotations = [], []
    cat_name_to_id = {}
    next_img_id, next_ann_id, next_cat_id = 1, 1, 1

    for jf in tqdm(sorted(json_files), desc="Merging JSON"):
        try:
            js = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:
            print("skip json parse error:", jf, "->", e)
            continue

        fname = js.get("file_name", js.get("image", js.get("filename")))
        ipath = resolve_image_path(fname, jf)
        if ipath is None:
            print("skip unresolved image:", jf)
            continue
        fname = ipath.name

        try:
            with Image.open(ipath) as im:
                W, H = im.size
        except Exception as e:
            print("skip image open error:", ipath, "->", e)
            continue

        # 객체 배열 탐색
        objs = []
        for k in ["annotations", "objects", "labels", "items", "annotation"]:
            if k in js and isinstance(js[k], list):
                objs = js[k]; break

        img_id = next_img_id
        images.append({"id": img_id, "file_name": fname, "width": int(W), "height": int(H)})
        next_img_id += 1

        for o in objs:
            cat = o.get("category_id", o.get("label", o.get("name")))
            if cat is None:
                attrs = o.get("attributes", {})
                if isinstance(attrs, dict):
                    cat = attrs.get("class")
            if cat is None:
                continue
            cname = str(cat)
            if cname not in cat_name_to_id:
                cat_name_to_id[cname] = next_cat_id
                next_cat_id += 1

            bb = norm_bbox(o.get("bbox", o.get("box", o.get("bndbox"))))
            if bb is None:
                continue
            x, y, w, h = bb
            annotations.append({
                "id": next_ann_id,
                "image_id": img_id,
                "category_id": cat_name_to_id[cname],
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w) * float(h),
                "iscrowd": 0
            })
            next_ann_id += 1

    categories = [{"id": v, "name": k} for k, v in sorted(cat_name_to_id.items(), key=lambda x: x[1])]
    coco = {"images": images, "annotations": annotations, "categories": categories}
    MERGED_COCO_JSON.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")

    print("merged file:", MERGED_COCO_JSON)
    print("images:", len(images), "annotations:", len(annotations), "categories:", len(categories))

if __name__ == "__main__":
    main()