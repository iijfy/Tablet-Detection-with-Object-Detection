import os, csv, random, shutil, yaml
from collections import defaultdict

PROJ = os.getcwd()
CSV  = "yolo/annotations_all.csv"
IMG_SRC = "/mnt/nas/jayden_code/ai05-level1-project/train_images"
OUT = os.path.join(PROJ, "data/yolo")
SPLIT = 0.8
SEED = 42

def ensure_dirs():
    for s in ["train","val"]:
        os.makedirs(os.path.join(OUT, s, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT, s, "labels"), exist_ok=True)

def load_rows():
    rows = []
    with open(CSV, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    assert rows, "CSV가 비었습니다."
    return rows

def build_cat_mapping(rows):
    id2name = {}
    for r in rows:
        oid = int(r["orig_cat_id"])
        id2name[oid] = r["class_name"]
    sorted_ids = sorted(id2name)
    old2new = {oid:i for i, oid in enumerate(sorted_ids)}
    names = [id2name[oid] for oid in sorted_ids]
    return old2new, names

def clip_bbox(x, y, w, h, W, H):
    # COCO 형식 x,y,w,h를 이미지 경계로 클리핑
    x = max(0.0, x); y = max(0.0, y)
    w = max(0.0, w); h = max(0.0, h)
    if x + w > W: w = max(0.0, W - x)
    if y + h > H: h = max(0.0, H - y)
    return x, y, w, h

def yolo_line(x, y, w, h, W, H, cls):
    # 정규화 (0~1) 보장
    cx = (x + w/2.0) / W
    cy = (y + h/2.0) / H
    nw = w / W
    nh = h / H
    # 수치적 오차 보정
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

def main():
    random.seed(SEED)
    ensure_dirs()
    rows = load_rows()
    old2new, names = build_cat_mapping(rows)

    # 이미지 단위로 묶기
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["filename"]].append(r)

    imgs = sorted(by_img.keys())
    random.shuffle(imgs)
    n_tr = int(len(imgs) * SPLIT)
    tr_set = set(imgs[:n_tr])

    skipped = 0
    for fname, annos in by_img.items():
        split = "train" if fname in tr_set else "val"
        src = os.path.join(IMG_SRC, fname)
        dst_img = os.path.join(OUT, split, "images", fname)
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        shutil.copy2(src, dst_img)

        lab_path = os.path.join(OUT, split, "labels", os.path.splitext(fname)[0] + ".txt")
        with open(lab_path, "w", encoding="utf-8") as f:
            for a in annos:
                W = float(a["img_w"]); H = float(a["img_h"])
                x = float(a["x"]);     y = float(a["y"])
                w = float(a["w"]);     h = float(a["h"])
                x, y, w, h = clip_bbox(x, y, w, h, W, H)
                if w <= 0 or h <= 0:
                    skipped += 1
                    continue
                cid = old2new[int(a["orig_cat_id"])]
                f.write(yolo_line(x, y, w, h, W, H, cid) + "\n")

    data_yaml = {
        "train": os.path.join(PROJ, "data/yolo/train"),
        "val":   os.path.join(PROJ, "data/yolo/val"),
        "test":  os.path.join(PROJ, "data/yolo/test"),
        "names": names
    }
    with open("yolo/data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"✅ 완료: data/yolo/*  (스킵된 박스 {skipped}개)")
    print(f"✅ data.yaml 생성 (클래스 {len(names)}개)")

if __name__ == "__main__":
    main()