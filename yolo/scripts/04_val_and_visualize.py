# yolo/scripts/04_val_and_visualize.py
# Validate on val split and save visualized predictions safely (OOM-safe, Korean text supported)

import os
import csv
import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# -------- paths --------
HERE = Path(__file__).resolve().parent
YOLO_ROOT = HERE.parent
DATA_ROOT = YOLO_ROOT / "data_clean"
VAL_IMAGES = DATA_ROOT / "val" / "images"
CLASSMAP_CSV = YOLO_ROOT / "metadata" / "class_map.csv"
DATA_YAML = YOLO_ROOT / "data.yaml"

def load_classmap(csv_path: Path):
    """
    class_map.csv -> (orig_cat_id -> yolo_id), (yolo_id -> (orig_cat_id, class_name))
    CSV columns must include: orig_cat_id, yolo_id, class_name
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"class_map.csv not found: {csv_path}")

    # utf-8-sig 로 읽어 BOM 이슈 방지
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)

    required = {"orig_cat_id", "yolo_id", "class_name"}
    if not required.issubset(set(rd.fieldnames or [])):
        raise ValueError(
            f"class_map.csv must have columns {sorted(list(required))} "
            f"(got: {rd.fieldnames})"
        )

    id2yolo = {}
    yolo2info = {}
    for r in rows:
        oc = int(r["orig_cat_id"])
        yi = int(r["yolo_id"])
        nm = str(r["class_name"])
        id2yolo[oc] = yi
        yolo2info[yi] = (oc, nm)
    return id2yolo, yolo2info

def try_load_font(font_path: str | None = None, size: int = 22):
    """
    한국어 지원 폰트 시도. 없으면 None 반환해서 ASCII fallback 사용.
    """
    candidates = []
    if font_path:
        candidates.append(font_path)
    # 리눅스 환경에서 흔한 위치들
    candidates += [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return None

def draw_box_with_text(img: Image.Image, xyxy, text: str, font: ImageFont.ImageFont | None):
    """
    PIL로 박스 + 텍스트 (한글 폰트 있으면 한글, 없으면 그대로 전달된 텍스트)
    xyxy: [x1,y1,x2,y2]
    """
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = map(int, xyxy)
    # box
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
    # text bg
    if not text:
        return img
    if font:
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    else:
        # font 없으면 기본 폰트(영문만 정상) + 대략적 높이
        tw, th = draw.textlength(text), 18
    pad = 4
    draw.rectangle([x1, max(0, y1 - th - pad*2), x1 + tw + pad*2, y1], fill=(0, 255, 0))
    if font:
        draw.text((x1 + pad, y1 - th - pad), text, fill=(0, 0, 0), font=font)
    else:
        draw.text((x1 + pad, y1 - th - pad), text, fill=(0, 0, 0))
    return img

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=str(YOLO_ROOT / "runs" / "yolov8l_baseline" / "weights" / "best.pt"),
                        help="path to best.pt")
    parser.add_argument("--data", type=str, default=str(DATA_YAML), help="data.yaml (absolute or relative)")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | 0 | 0,1 ...")
    parser.add_argument("--limit", type=int, default=24, help="num of validation images to visualize (per-image inference)")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--font", type=str, default=None, help="optional TTF/TTC path for Korean")
    args = parser.parse_args()

    # 출력 폴더
    out_dir = YOLO_ROOT / f"visualized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    (out_dir).mkdir(parents=True, exist_ok=True)

    # class map 로드
    _, yolo2info = load_classmap(CLASSMAP_CSV)

    # 모델 로드 (+ device auto fallback)
    def _load_on(device_str: str):
        print(f"[INFO] Loading model on device={device_str} ...")
        m = YOLO(args.weights)
        # val()를 직접 쓰지 않고 per-image predict로 OOM 예방
        return m, device_str

    if args.device == "auto":
        try:
            model, use_dev = _load_on("0")
            # 빠르게 warm-up
            model.predict(source=[str(next(VAL_IMAGES.glob("*.png")))], imgsz=args.imgsz, device=use_dev, conf=args.conf, verbose=False)
        except Exception as e:
            print(f"[WARN] GPU load/predict failed → falling back to CPU. ({e})")
            model, use_dev = _load_on("cpu")
    else:
        model, use_dev = _load_on(args.device)

    # 폰트 로드 (한글 지원 시도)
    font = try_load_font(args.font, size=22)
    if font is None:
        print("[WARN] Korean font not found. Falling back to ASCII-safe labels (category_id).")

    # val 이미지 리스트
    img_paths = sorted([p for p in VAL_IMAGES.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if args.limit and args.limit > 0:
        img_paths = img_paths[:args.limit]

    print(f"[INFO] Val images: {len(img_paths)}")
    print(f"[INFO] Saving to: {out_dir}")

    # 한 장씩 예측 → 즉시 시각화 → 저장 (GPU 메모리 안전)
    for i, ip in enumerate(img_paths, 1):
        try:
            rs = model.predict(
                source=str(ip),
                imgsz=args.imgsz,
                device=use_dev,
                conf=args.conf,
                verbose=False
            )
        except RuntimeError as e:
            # OOM 등 발생 시 CPU로 재시도
            if "out of memory" in str(e).lower():
                print(f"[WARN] OOM on GPU for {ip.name}. Retrying on CPU...")
                rs = model.predict(source=str(ip), imgsz=args.imgsz, device="cpu", conf=args.conf, verbose=False)
            else:
                raise

        # 결과 파싱
        r = rs[0]
        # 원본 로드
        img = Image.open(ip).convert("RGB")

        if r.boxes is not None and len(r.boxes) > 0:
            # xyxy, cls, conf
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()

            for b, c, s in zip(xyxy, cls, conf):
                orig_id, name_kor = yolo2info.get(c, (c, str(c)))
                # 폰트 없으면 카테고리ID만 표기 (ASCII-safe). 폰트 있으면 한글+ID 같이 표기.
                if font is None:
                    label = f"{orig_id} {s:.2f}"
                else:
                    label = f"{name_kor} ({orig_id}) {s:.2f}"
                img = draw_box_with_text(img, b, label, font)
        else:
            # no dets → 그대로 저장
            pass

        save_path = out_dir / ip.name
        img.save(save_path)
        if i % 10 == 0 or i == len(img_paths):
            print(f"[INFO] {i}/{len(img_paths)} saved")

    print(f"[DONE] Visualizations saved in: {out_dir}")
    print("Tip) If you still see ASCII '???', install a Korean TTF and pass --font /path/to/NanumGothic.ttf")

if __name__ == "__main__":
    main()