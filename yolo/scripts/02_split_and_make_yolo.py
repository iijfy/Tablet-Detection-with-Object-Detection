# -*- coding: utf-8 -*-
# 8:2 분할 + YOLO 라벨/이미지 생성 (복사 방식, 안전 클리핑 포함)

import os, csv, json, glob, random, shutil
from collections import defaultdict

# ===== 경로 설정 =====
ROOT = os.path.dirname(os.path.dirname(__file__))                 # .../yolo
ANN_JSON_DIR = "/mnt/nas/jayden_code/ai05-level1-project/train_annotations"
IMG_SRC_DIR  = "/mnt/nas/jayden_code/ai05-level1-project/train_images"

DATA_ROOT    = os.path.join(ROOT, "data_clean")                   # 최종 YOLO 데이터 루트
IMGS_TRAIN   = os.path.join(DATA_ROOT, "train/images")
IMGS_VAL     = os.path.join(DATA_ROOT, "val/images")
LBL_TRAIN    = os.path.join(DATA_ROOT, "train/labels")
LBL_VAL      = os.path.join(DATA_ROOT, "val/labels")

CLASSMAP_CSV = os.path.join(ROOT, "metadata/class_map.csv")       # category_id,class_name (1단계 산출물)
DATA_YAML    = os.path.join(ROOT, "data.yaml")                    # (1단계 산출물)

SPLIT_RATIO  = 0.8
SPLIT_SEED   = 42

def info(*a): print("[INFO]", *a)
def warn(*a): print("[WARN]", *a)

def ensure_dirs():
    for p in [IMGS_TRAIN, IMGS_VAL, LBL_TRAIN, LBL_VAL]:
        os.makedirs(p, exist_ok=True)

def read_classmap_csv():
    """
    class_map.csv를 유연하게 파싱:
      - id 컬럼: ['category_id','orig_cat_id','id','cat_id'] 중 1개
      - name 컬럼: ['class_name','name','label'] 중 1개
      - 둘 다 못 찾으면 첫 번째 컬럼을 id, 두 번째 컬럼을 name으로 가정
    """
    if not os.path.exists(CLASSMAP_CSV):
        raise SystemExit(f"class_map.csv 없음: {CLASSMAP_CSV}")

    # 후보 컬럼 이름
    id_keys = ["category_id", "orig_cat_id", "id", "cat_id"]
    name_keys = ["class_name", "name", "label"]

    id2name = {}

    with open(CLASSMAP_CSV, "r", encoding="utf-8") as f:
        rd = csv.reader(f)
        rows = list(rd)
        if not rows:
            raise SystemExit(f"class_map.csv가 비어있습니다: {CLASSMAP_CSV}")

        header = [h.strip() for h in rows[0]]
        data = rows[1:] if len(rows) > 1 else []

        # 헤더를 DictReader처럼 매핑 시도
        def find_idx(keys, default=None):
            for k in keys:
                if k in header:
                    return header.index(k)
            return default

        id_idx = find_idx(id_keys)
        name_idx = find_idx(name_keys)

        # 헤더 매칭 실패 시, 1열=ID, 2열=NAME 가정
        if id_idx is None or name_idx is None:
            if len(header) >= 2:
                id_idx = 0 if id_idx is None else id_idx
                name_idx = 1 if name_idx is None else name_idx
            else:
                raise SystemExit(
                    f"class_map.csv 헤더를 해석할 수 없습니다. "
                    f"필요 헤더 예시: 'category_id,class_name'"
                )

        for r in data:
            if not r or len(r) <= max(id_idx, name_idx):
                continue
            try:
                cid = int(str(r[id_idx]).strip())
            except Exception:
                continue
            cname = str(r[name_idx]).strip()
            id2name[cid] = cname

    if not id2name:
        raise SystemExit("class_map.csv에서 id/name을 읽지 못했습니다. 파일 형식 확인.")

    sorted_ids = sorted(id2name.keys())
    id2yolo = {cid: i for i, cid in enumerate(sorted_ids)}
    info(f"class_map 로드 완료: classes={len(sorted_ids)} (헤더={header})")
    return id2yolo

def load_annotations_from_jsons(json_dir):
    """
    이미지별 어노테이션 수집:
      - img_boxes[fname] = [(x,y,w,h, orig_cat_id), ...]
      - img_size[fname]  = (W, H)
    """
    img_boxes = defaultdict(list)
    img_size  = {}

    json_files = glob.glob(os.path.join(json_dir, "**/*.json"), recursive=True)
    assert json_files, f"JSON을 찾지 못함: {json_dir}"

    for jp in json_files:
        try:
            d = json.load(open(jp, "r", encoding="utf-8"))
        except Exception as e:
            warn(f"JSON 로드 실패: {jp} -> {e}")
            continue

        # id -> (file_name, W, H)
        imap = {}
        for im in d.get("images", []):
            try:
                imap[int(im["id"])] = (im["file_name"], int(im["width"]), int(im["height"]))
            except Exception:
                continue

        for a in d.get("annotations", []):
            if "bbox" not in a or "image_id" not in a or "category_id" not in a:
                continue
            x, y, w, h = a["bbox"]
            img_id = int(a["image_id"])
            if img_id not in imap:
                continue
            fname, W, H = imap[img_id]

            # 상자 클리핑 (이미지 경계 밖 숫자 방지)
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(W), x1 + float(w))
            y2 = min(float(H), y1 + float(h))
            cw = max(0.0, x2 - x1)
            ch = max(0.0, y2 - y1)
            if cw <= 0 or ch <= 0:
                continue  # 완전히 밖이면 버림

            cid = int(a["category_id"])
            img_boxes[fname].append((x1, y1, cw, ch, cid))
            img_size[fname] = (W, H)

    info(f"어노테이션 로드 완료: images={len(img_size)}")
    return img_boxes, img_size

def split_files(fnames, ratio=0.8, seed=42):
    lst = list(fnames)
    random.Random(seed).shuffle(lst)
    n_tr = int(len(lst) * ratio)
    return set(lst[:n_tr]), set(lst[n_tr:])

def save_yolo_label(txt_path, boxes, W, H, id2yolo):
    lines = []
    for x, y, w, h, orig_id in boxes:
        # YOLO format: class cx cy nw nh  (모두 0~1 정규화)
        cx = (x + w / 2.0) / float(W)
        cy = (y + h / 2.0) / float(H)
        nw = w / float(W)
        nh = h / float(H)

        # 경계 살짝 클립 (숫자 미세 오차로 1.0000001 같은 것 방지)
        def clip01(v): return max(0.0, min(1.0, v))
        cx, cy, nw, nh = map(clip01, (cx, cy, nw, nh))

        if orig_id not in id2yolo:
            warn(f"미정의 category_id 발견: {orig_id} -> 라벨 스킵")
            continue
        cls = id2yolo[orig_id]
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

def copy_image(src_fname, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    src = os.path.join(IMG_SRC_DIR, src_fname)
    dst = os.path.join(dst_dir, src_fname)
    if not os.path.exists(src):
        warn(f"원본 이미지 없음: {src}")
        return
    shutil.copy2(src, dst)

def main():
    ensure_dirs()
    if not os.path.exists(DATA_YAML):
        warn(f"data.yaml 미존재: {DATA_YAML}  (1단계 먼저 실행 필요)")
    id2yolo = read_classmap_csv()
    img_boxes, img_size = load_annotations_from_jsons(ANN_JSON_DIR)
    if not img_boxes:
        raise SystemExit("어노테이션이 비었습니다. JSON 경로/형식 확인.")

    train_set, val_set = split_files(img_boxes.keys(), SPLIT_RATIO, SPLIT_SEED)
    info(f"Split: train={len(train_set)}  val={len(val_set)}")

    # 저장
    for fname in train_set:
        if fname not in img_size: 
            continue
        W, H = img_size[fname]
        copy_image(fname, IMGS_TRAIN)
        save_yolo_label(
            os.path.join(LBL_TRAIN, os.path.splitext(fname)[0] + ".txt"),
            img_boxes[fname], W, H, id2yolo
        )

    for fname in val_set:
        if fname not in img_size: 
            continue
        W, H = img_size[fname]
        copy_image(fname, IMGS_VAL)
        save_yolo_label(
            os.path.join(LBL_VAL, os.path.splitext(fname)[0] + ".txt"),
            img_boxes[fname], W, H, id2yolo
        )

    info(f"✅ YOLO 데이터셋 생성 완료: {DATA_ROOT}")
    info(f" - train images: {len(os.listdir(IMGS_TRAIN))}  | labels: {len(os.listdir(LBL_TRAIN))}")
    info(f" - val   images: {len(os.listdir(IMGS_VAL))}    | labels: {len(os.listdir(LBL_VAL))}")

if __name__ == "__main__":
    main()