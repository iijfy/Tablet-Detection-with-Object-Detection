# -*- coding: utf-8 -*-
"""
데이터 중복/누수 진단 리포트 생성기 😊
- 완전 중복: MD5 해시로 동일 이미지 그룹 찾기
- 근사 중복: aHash(8x8 평균 해시) 해밍거리로 유사 후보 찾기
- 라벨 중복: 동일 이미지 내 IoU >= 0.95 박스 쌍(같은 category)
- 스플릿 누수: train/val/test 사이에 동일 이미지(해시) 존재
출력: CSV 여러 개 + 요약 출력
"""

import os, sys, csv, json, glob, hashlib
from collections import defaultdict
from PIL import Image

# ========= 사용자 환경 경로 설정 (네 환경 반영) =========
ANN_JSON_DIR = "/mnt/nas/jayden_code/ai05-level1-project/train_annotations"  # COCO json들(하위 폴더 재귀)
IMG_TRAIN_DIR = "/mnt/nas/jayden_code/ai05-level1-project/train_images"      # 원본 train 이미지 루트
IMG_TEST_DIR  = "/mnt/nas/jayden_code/ai05-level1-project/test_images"       # 원본 test 이미지 루트

# (선택) YOLO로 변환된 최종 split 경로(있다면 누수 점검에 활용)
YOLO_TRAIN_IMG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_clean/train/images")
YOLO_VAL_IMG   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_clean/val/images")

CLASSMAP_CSV   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metadata/class_map.csv")
OUT_DIR        = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")  # 리포트 저장 폴더

# ========= 유틸 =========
def info(*a): print("[INFO]", *a)
def warn(*a): print("[WARN]", *a)
os.makedirs(OUT_DIR, exist_ok=True)

def md5_of_file(path, chunk=8192):
    """파일의 MD5 해시 계산해요 🔐"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def average_hash(img_path, size=8):
    """aHash 만들기 🧩: 작게(8x8) 줄이고 평균보다 큰 픽셀=1, 작으면=0 비트"""
    try:
        with Image.open(img_path) as im:
            im = im.convert("L").resize((size, size))  # 그레이스케일 + 8x8
            pixels = list(im.getdata())
    except Exception:
        return None  # 열기 실패
    avg = sum(pixels) / len(pixels)
    bits = [1 if p >= avg else 0 for p in pixels]
    # 64비트를 16진 문자열로
    val = 0
    for b in bits:
        val = (val << 1) | b
    return f"{val:016x}"

def hamming_hex(h1, h2):
    """16진 해시 문자열 사이 해밍 거리(비트 단위) 📏"""
    if (h1 is None) or (h2 is None):
        return 999
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")

def load_coco_annotations(json_root):
    """
    COCO json들을 모두 읽어 이미지별 라벨 수집해요 📚
    반환:
      img_boxes[fname] = [(x,y,w,h,cid), ...]
      img_size[fname]  = (W,H)
      class_ids        = set(cid)
    """
    img_boxes = defaultdict(list)
    img_size  = {}
    class_ids = set()

    json_files = glob.glob(os.path.join(json_root, "**/*.json"), recursive=True)
    assert json_files, f"JSON이 없습니다: {json_root}"

    for jp in json_files:
        try:
            d = json.load(open(jp, "r", encoding="utf-8"))
        except Exception as e:
            warn(f"JSON 로드 실패: {jp} -> {e}")
            continue

        imap = {}
        for im in d.get("images", []):
            try:
                _id = int(im["id"])
                fname = im["file_name"]
                W = int(im["width"]); H=int(im["height"])
                imap[_id] = (fname, W, H)
            except Exception:
                continue

        for a in d.get("annotations", []):
            if "bbox" not in a or "image_id" not in a or "category_id" not in a:
                continue
            x,y,w,h = a["bbox"]
            img_id = int(a["image_id"])
            if img_id not in imap: 
                continue
            fname,W,H = imap[img_id]
            # 클리핑 (음수/경계 초과 방지)
            x1 = max(0.0, float(x)); y1 = max(0.0, float(y))
            x2 = min(float(W), x1 + float(w))
            y2 = min(float(H), y1 + float(h))
            cw = max(0.0, x2 - x1); ch = max(0.0, y2 - y1)
            if cw <= 0 or ch <= 0: 
                continue

            cid = int(a["category_id"])
            img_boxes[fname].append((x1,y1,cw,ch,cid))
            img_size[fname] = (W,H)
            class_ids.add(cid)

    return img_boxes, img_size, class_ids

def iou_xywh(b1, b2):
    """IoU 계산기 📐: (x,y,w,h)"""
    x1,y1,w1,h1 = b1
    x2,y2,w2,h2 = b2
    xa1, ya1 = x1, y1
    xa2, ya2 = x1+w1, y1+h1
    xb1, yb1 = x2, y2
    xb2, yb2 = x2+w2, y2+h2
    inter_w = max(0, min(xa2, xb2) - max(xa1, xb1))
    inter_h = max(0, min(ya2, yb2) - max(ya1, yb1))
    inter = inter_w * inter_h
    area1 = w1*h1; area2 = w2*h2
    union = area1 + area2 - inter
    if union <= 0: return 0.0
    return inter / union

# ========= 1) 완전중복/근사중복 탐지 =========
def scan_images_md5_ahash(root_dir):
    """루트 아래 모든 이미지의 MD5/aHash 스캔 🔎"""
    exts = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
    paths = []
    for r,_,fs in os.walk(root_dir):
        for f in fs:
            if os.path.splitext(f.lower())[1] in exts:
                paths.append(os.path.join(r,f))

    md5_map = defaultdict(list)
    ahash_map = {}
    for p in paths:
        try:
            m = md5_of_file(p)
        except Exception as e:
            warn(f"MD5 실패: {p} -> {e}")
            continue
        md5_map[m].append(p)
        ahash_map[p] = average_hash(p)
    return md5_map, ahash_map

# ========= 2) 스플릿 누수 점검 =========
def leak_between_splits(split_dirs):
    """
    split_dirs: [('train', dir1), ('val', dir2), ('test', dir3)]
    동일 MD5 해시가 여러 split에 존재하면 누수 후보 ⚠️
    """
    tag_of = {}  # md5 -> set(tags)
    file_of = defaultdict(list)  # md5 -> [paths]
    for tag, d in split_dirs:
        if not d or not os.path.exists(d): 
            continue
        md5_map, _ = scan_images_md5_ahash(d)
        for h, paths in md5_map.items():
            tag_of.setdefault(h, set()).add(tag)
            file_of[h].extend(paths)
    leaks = []
    for h, tags in tag_of.items():
        if len(tags) >= 2:
            leaks.append((h, ",".join(sorted(tags)), len(file_of[h]), file_of[h][:10]))
    return leaks

# ========= 3) 라벨 중복 점검 =========
def label_duplicates(img_boxes, thr=0.95):
    """
    같은 이미지 내에서 같은 category 간 IoU>=thr 인 박스 쌍을 찾는다 📎
    """
    rows = []
    for fname, boxes in img_boxes.items():
        n = len(boxes)
        for i in range(n):
            x1,y1,w1,h1,c1 = boxes[i]
            for j in range(i+1, n):
                x2,y2,w2,h2,c2 = boxes[j]
                if c1 != c2: 
                    continue
                iou = iou_xywh((x1,y1,w1,h1), (x2,y2,w2,h2))
                if iou >= thr:
                    rows.append([fname, i, j, c1, round(iou,4)])
    return rows

# ========= 4) 클래스 분포 표 =========
def class_distribution(img_boxes):
    cnt = defaultdict(int)
    for fname, boxes in img_boxes.items():
        for _,_,_,_,cid in boxes:
            cnt[cid]+=1
    rows = sorted(cnt.items(), key=lambda x: x[0])
    return rows

# ========= 메인 =========
def main():
    info("COCO 주석 로드 중...")
    img_boxes, img_size, class_ids = load_coco_annotations(ANN_JSON_DIR)
    info(f"라벨 이미지 수: {len(img_boxes)} | 클래스 수: {len(class_ids)}")

    # (A) 원본 train/test 이미지 중복
    info("원본 train_images 스캔(완전/근사 중복)...")
    md5_train, ahash_train = scan_images_md5_ahash(IMG_TRAIN_DIR)
    info("원본 test_images 스캔(완전/근사 중복)...")
    md5_test, ahash_test   = scan_images_md5_ahash(IMG_TEST_DIR)

    # 완전 중복 그룹(원본 train 내)
    dup_exact_rows = []
    for h, paths in md5_train.items():
        if len(paths) >= 2:
            dup_exact_rows.append([h, len(paths)] + paths[:10])
    with open(os.path.join(OUT_DIR, "duplicate_exact_train.csv"), "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f); cw.writerow(["md5","count","sample_paths_up_to_10"])
        for r in dup_exact_rows: cw.writerow(r)

    # 근사 중복 후보(원본 train 내) — 해밍거리 <= 5
    # 간단히 상위 N개만 페어링(전체 페어 O(N^2) 방지하려면 해시버킷/그리드가 좋음)
    paths = list(ahash_train.keys())
    approx_rows = []
    N = len(paths)
    LIM = 5000  # 너무 크면 부분 샘플만 비교
    step_paths = paths[:min(N, LIM)]
    for i in range(len(step_paths)):
        for j in range(i+1, len(step_paths)):
            d = hamming_hex(ahash_train[step_paths[i]], ahash_train[step_paths[j]])
            if d <= 5:
                approx_rows.append([d, step_paths[i], step_paths[j]])
    with open(os.path.join(OUT_DIR, "duplicate_ahash_candidates_train.csv"), "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f); cw.writerow(["hamming_distance","path_a","path_b"])
        cw.writerows(sorted(approx_rows, key=lambda r:r[0]))

    # (B) 스플릿 누수 (YOLO 변환 split 기준 + 원본 train/test 기준 모두 점검)
    leaks = leak_between_splits([
        ("yolo_train", YOLO_TRAIN_IMG if os.path.exists(YOLO_TRAIN_IMG) else None),
        ("yolo_val",   YOLO_VAL_IMG   if os.path.exists(YOLO_VAL_IMG)   else None),
        ("orig_train", IMG_TRAIN_DIR),
        ("orig_test",  IMG_TEST_DIR),
    ])
    with open(os.path.join(OUT_DIR, "split_leakage.csv"), "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f); cw.writerow(["md5","splits","count","sample_paths_up_to_10"])
        for h, tags, c, sample in leaks:
            cw.writerow([h, tags, c, " | ".join(sample)])

    # (C) 라벨 중복 (동일 이미지 내부 IoU>=0.95 쌍)
    rows = label_duplicates(img_boxes, thr=0.95)
    with open(os.path.join(OUT_DIR, "label_duplicates_iou095.csv"), "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f); cw.writerow(["image_file","box_idx_a","box_idx_b","category_id","iou"])
        cw.writerows(rows)

    # (D) 클래스 분포 표
    dist = class_distribution(img_boxes)
    with open(os.path.join(OUT_DIR, "class_distribution_overall.csv"), "w", newline="", encoding="utf-8") as f:
        cw = csv.writer(f); cw.writerow(["category_id","count"])
        cw.writerows(dist)

    # (E) 요약 프린트
    n_exact_groups = sum(1 for _,paths in md5_train.items() if len(paths)>=2)
    info("===== 요약 =====")
    info(f"완전 중복 그룹 수(train): {n_exact_groups}")
    info(f"근사 중복 후보(train, aHash<=5, max {len(step_paths)}개 비교): {len(approx_rows)}")
    info(f"스플릿 누수 후보 개수: {len(leaks)}")
    info(f"라벨 중복 쌍(IoU>=0.95): {len(rows)}")
    info(f"클래스 분포 파일: {os.path.join(OUT_DIR,'class_distribution_overall.csv')}")
    info(f"리포트 폴더: {OUT_DIR}")

if __name__ == "__main__":
    main()