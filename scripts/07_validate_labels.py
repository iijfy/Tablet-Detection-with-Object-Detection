# scripts/07_validate_labels.py
from pathlib import Path
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from configs.paths import DATASET_DIR, IMG_DIR

# 경로 상수
LABEL_TRAIN = DATASET_DIR / "labels" / "train"  # YOLO 라벨 경로
LABEL_VAL   = DATASET_DIR / "labels" / "val"    # YOLO 라벨 경로
IMG_TRAIN   = IMG_DIR / "train"                 # 학습 이미지 경로
IMG_VAL     = IMG_DIR / "val"                   # 검증 이미지 경로
OUT_DIR     = (Path(__file__).resolve().parents[1] / "runs" / "viz" / "label_check")

def read_yolo_label(txt_path: Path):
    """YOLO txt 라벨 파싱"""
    lines = txt_path.read_text(encoding="utf-8").strip().splitlines() if txt_path.exists() else []
    items = []
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = parts
        items.append((int(cls), float(cx), float(cy), float(w), float(h)))
    return items

def draw_boxes(img_path: Path, labels, out_path: Path):
    """그림 저장"""
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(im); ax.axis("off"); ax.set_title(f"{img_path.name}")
    for cls, cx, cy, w, h in labels:
        x = (cx - w / 2.0) * W
        y = (cy - h / 2.0) * H
        ww = w * W
        hh = h * H
        rect = patches.Rectangle((x, y), ww, hh, linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 2, str(cls), color="lime", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def sample_and_visualize(split_name: str, img_dir: Path, label_dir: Path, k: int = 3):
    """무작위 샘플 시각화"""
    imgs = [p for p in img_dir.glob("*") if p.is_file()]
    if not imgs:
        print(f"{split_name} 이미지 없음")
        return
    random.seed(42)
    picks = random.sample(imgs, k=min(k, len(imgs)))
    for ip in picks:
        lp = label_dir / f"{ip.stem}.txt"
        labels = read_yolo_label(lp)
        outp = OUT_DIR / split_name / f"{ip.stem}_check.jpg"
        draw_boxes(ip, labels, outp)
        print("saved:", outp)

def main():
    sample_and_visualize("train", IMG_TRAIN, LABEL_TRAIN, k=3)
    sample_and_visualize("val",   IMG_VAL,   LABEL_VAL,   k=3)

if __name__ == "__main__":
    main()