import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === 입력 CSV ===
CSV_PATH = "yolo/annotations.csv"

# === 원본 이미지 경로 (NAS) ===
IMG_DIR = "/mnt/nas/jayden_code/ai05-level1-project/train_images"

# === 출력 폴더 ===
OUT_DIR = "yolo/data_clean"
TRAIN_RATIO = 0.8

def make_symlink(img_name, split):
    """원본 이미지를 복사하여 저장"""
    src = os.path.join(IMG_DIR, img_name)
    dst_dir = os.path.join(OUT_DIR, split, "images")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, img_name)
    if not os.path.exists(dst):
        try:
            shutil.copy2(src, dst)  # symlink 대신 복사
        except FileNotFoundError:
            print(f"[WARN] 원본 이미지를 찾을 수 없음: {src}")

def make_label(row, split):
    """YOLO 포맷 txt 생성"""
    label_dir = os.path.join(OUT_DIR, split, "labels")
    os.makedirs(label_dir, exist_ok=True)

    fname = os.path.splitext(os.path.basename(row["filename"]))[0]
    label_path = os.path.join(label_dir, f"{fname}.txt")

    # YOLO format = class_id cx cy w h (normalized)
    x, y, w, h = row[["x", "y", "w", "h"]]
    cx = x + w / 2
    cy = y + h / 2
    nx = cx / row["img_w"]
    ny = cy / row["img_h"]
    nw = w / row["img_w"]
    nh = h / row["img_h"]

    with open(label_path, "a") as f:
        f.write(f"{row['orig_cat_id']} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")

def main():
    df = pd.read_csv(CSV_PATH)
    print(f"총 데이터 수: {len(df)}")

    # train / val 분할
    train_df, val_df = train_test_split(df, test_size=1-TRAIN_RATIO, random_state=42)
    splits = {"train": train_df, "val": val_df}

    for split, data in splits.items():
        for _, row in data.iterrows():
            make_symlink(row["filename"], split)
            make_label(row, split)
        print(f"✅ {split} 세트 구성 완료: {len(data)}개 어노테이션")

    print("\n🎯 YOLO 데이터셋 생성 완료:")
    print(f" - train 폴더: {OUT_DIR}/train/images, labels")
    print(f" - val 폴더:   {OUT_DIR}/val/images, labels")

if __name__ == "__main__":
    main()