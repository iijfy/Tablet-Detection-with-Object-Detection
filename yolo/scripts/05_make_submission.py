# yolo/scripts/05_make_submission.py
import os, re, csv, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
YOLO_ROOT = HERE.parent

CLASSMAP_CSV = YOLO_ROOT / "metadata" / "class_map.csv"
DATA_YAML    = YOLO_ROOT / "data.yaml"
TEST_DIR     = Path("/mnt/nas/jayden_code/ai05-level1-project/test_images")  # 필요 시 --test_dir 로 바꿀 수 있음
OUT_CSV_DEF  = YOLO_ROOT / "submission.csv"

def load_yolo2orig(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        need = {"orig_cat_id", "yolo_id", "class_name"}
        if not need.issubset(set(rd.fieldnames or [])):
            raise ValueError(f"class_map.csv 컬럼에 {sorted(list(need))} 가 필요합니다. 현재: {rd.fieldnames}")
        yolo2orig = {}
        for r in rd:
            yolo2orig[int(r["yolo_id"])] = int(r["orig_cat_id"])
    return yolo2orig

def extract_image_id(fname: str):
    """파일명에서 '마지막 숫자 덩어리'를 image_id 로 사용"""
    m = re.findall(r"\d+", Path(fname).stem)
    return int(m[-1]) if m else None

def clamp_int(v, lo, hi):
    return int(max(lo, min(hi, round(float(v)))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str,
                    default=str(YOLO_ROOT / "runs" / "yolov8l_baseline" / "weights" / "best.pt"))
    ap.add_argument("--test_dir", type=str, default=str(TEST_DIR))
    ap.add_argument("--out_csv", type=str, default=str(OUT_CSV_DEF))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|0|0,1")
    args = ap.parse_args()

    yolo2orig = load_yolo2orig(CLASSMAP_CSV)

    # 모델 로드 (auto→GPU 실패 시 CPU 폴백)
    def _predict_once(model, img_path, device):
        return model.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf,
                             device=device, verbose=False)[0]

    print(f"[INFO] weights: {args.weights}")
    print(f"[INFO] test_dir: {args.test_dir}")
    print(f"[INFO] out_csv : {args.out_csv}")

    model = YOLO(args.weights)
    device_try = args.device
    # 간단 워밍업 시도
    try:
        some = next(iter(sorted(Path(args.test_dir).glob("*"))))
        _predict_once(model, some, device_try)
    except Exception as e:
        if args.device == "auto":
            print(f"[WARN] GPU 예열 실패 → CPU로 변경 ({e})")
            device_try = "cpu"
        else:
            raise

    test_imgs = sorted([p for p in Path(args.test_dir).glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    assert test_imgs, f"테스트 이미지가 없습니다: {args.test_dir}"

    rows = []
    ann_id = 1
    for p in test_imgs:
        iid = extract_image_id(p.name)
        if iid is None:
            print(f"[WARN] image_id 추출 실패 → 스킵: {p.name}")
            continue

        # 이미지 크기 (클램프용)
        with Image.open(p) as im:
            W, H = im.size

        try:
            r = _predict_once(model, p, device_try)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[WARN] OOM on GPU → CPU 재시도: {p.name}")
                r = _predict_once(model, p, "cpu")
            else:
                raise

        if r.boxes is None or len(r.boxes) == 0:
            continue

        xywh = r.boxes.xywh.cpu().numpy()  # 픽셀 기준
        cls  = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()

        for (cx, cy, w, h), c, s in zip(xywh, cls, conf):
            # xywh → 좌상단 원점 기준 xywh 로 변환 + 반올림/클램프
            x = clamp_int(cx - w / 2, 0, W - 1)
            y = clamp_int(cy - h / 2, 0, H - 1)
            bw = clamp_int(w, 1, W)       # 최소 1픽셀
            bh = clamp_int(h, 1, H)

            # 경계 넘지 않게 한 번 더 보정
            if x + bw > W: bw = W - x
            if y + bh > H: bh = H - y

            cat_id = yolo2orig.get(int(c), int(c))  # 매핑 없으면 보수적으로 c 사용
            rows.append([ann_id, iid, cat_id, x, y, bw, bh, round(float(s), 2)])
            ann_id += 1

    # 저장
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"])
        wr.writerows(rows)

    print(f"[OK] 제출 파일 저장: {out_csv}  (rows={len(rows)})")
    print("※ 캐글 업로드 전, 맨 위/아래 몇 줄을 꼭 확인해보세요.")
    print("  - image_id가 파일명 숫자와 일치하는지")
    print("  - bbox가 0~이미지크기 범위 안인지")
    print("  - category_id가 기대한 정수(원본 ID)인지")

if __name__ == "__main__":
    main()