#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train/Test 중복 이미지 탐지 스크립트
- 구조 해시(Perceptual Hash), 색상 히스토그램 코사인 유사도, SSIM/MSE 2단계 정밀 비교
- 결과를 TXT와 CSV로 모두 저장
- 기본 경로는 현재 프로젝트 폴더 구조에 맞춰 설정됨

Usage (기본값 그대로 실행):
  python tools/check_duplicates.py

경로/임계치 변경 실행 예:
  python tools/check_duplicates.py \
    --train_dir "/path/to/train/images" \
    --test_dir "/path/to/test/images" \
    --out_txt "duplicate_report.txt" \
    --out_csv "duplicate_pairs.csv" \
    --hash_threshold 6 --hist_threshold 0.94 \
    --ssim_threshold 0.985 --mse_threshold 0.0015
"""

import os
import csv
import argparse
from pathlib import Path

from PIL import Image
import imagehash
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".mpo", ".dng", ".pfm"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def load_image_rgb(path: str, target_size=(256, 256)) -> np.ndarray:
    # 이미지를 RGB로 로드하고 [0,1] float32 로 스케일
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(target_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def compute_hash_and_hist(path: str, hist_bins=16):
    # 구조적 특징: perceptual hash (밝기 기반)
    # 색 분포: 채널별 히스토그램 (정규화)
    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        h = imagehash.phash(img_rgb)

        arr = np.asarray(img_rgb, dtype=np.uint8)
        hist_feats = []
        for c in range(3):  # R,G,B
            hist, _ = np.histogram(
                arr[..., c],
                bins=hist_bins,
                range=(0, 256),
                density=True,
            )
            hist_feats.append(hist.astype(np.float32))
        hist_vec = np.concatenate(hist_feats)  # (hist_bins*3,)
    return h, hist_vec


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return dot / (na * nb + eps)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.mean(diff * diff))


def compare_pair(img_a_path: str, img_b_path: str, size_for_diff=(256, 256)):
    A = load_image_rgb(img_a_path, target_size=size_for_diff)
    B = load_image_rgb(img_b_path, target_size=size_for_diff)
    ssim_val = ssim(A, B, channel_axis=2, data_range=1.0)
    mse_val = mse(A, B)
    return ssim_val, mse_val


def build_index(folder: str, hist_bins=16):
    folder = Path(folder)
    assert folder.exists(), f"[ERROR] Not found: {folder}"
    index = []
    files = [p for p in folder.iterdir() if is_image_file(p)]
    for p in tqdm(files, desc=f"Indexing {folder.name}"):
        try:
            h, hist_vec = compute_hash_and_hist(str(p), hist_bins=hist_bins)
            index.append({
                "name": p.name,
                "path": str(p),
                "hash": h,
                "hist": hist_vec,
            })
        except Exception as e:
            print(f"[WARN] {p.name}: {e}")
    return index


def find_duplicates(
    train_dir: str,
    test_dir: str,
    hash_threshold=5,
    hist_threshold=0.95,
    ssim_threshold=0.98,
    mse_threshold=0.001,
    hist_bins=16,
):
    train_index = build_index(train_dir, hist_bins=hist_bins)
    test_index = build_index(test_dir, hist_bins=hist_bins)

    results = []
    # 빠른 조회를 위해 train_index 를 해시값 기준으로 버킷화 (대충 근처만 탐색)
    buckets = {}
    for tr in train_index:
        hv = int(str(tr["hash"]), 16)  # hash 객체를 int 로
        buckets.setdefault(hv, []).append(tr)

    def candidates_for(test_hash_obj, window=hash_threshold + 1):
        th = int(str(test_hash_obj), 16)
        # 해시값 근사 범위만 순회 (O(window^2)까지는 아니고 keys만 순회)
        for hv, items in buckets.items():
            if abs(hv - th) <= 100000000000000000000000000000000000:  # 그냥 키 제한 없이 두고, 실제 필터는 해밍거리로
                for it in items:
                    yield it

    for t_item in tqdm(test_index, desc="Cross-check train vs test"):
        # 1차: 해밍 거리 + 히스토그램 코사인 유사도로 거르고
        for tr_item in candidates_for(t_item["hash"]):
            hash_dist = abs(t_item["hash"] - tr_item["hash"])
            if hash_dist > hash_threshold:
                continue
            hist_sim = cosine_similarity(t_item["hist"], tr_item["hist"])
            if hist_sim < hist_threshold:
                continue
            # 2차: SSIM & MSE 정밀 비교
            ssim_val, mse_val = compare_pair(t_item["path"], tr_item["path"])
            if ssim_val >= ssim_threshold and mse_val <= mse_threshold:
                results.append({
                    "train_img": tr_item["name"],
                    "train_path": tr_item["path"],
                    "test_img": t_item["name"],
                    "test_path": t_item["path"],
                    "hash_dist": int(hash_dist),
                    "hist_sim": float(hist_sim),
                    "ssim": float(ssim_val),
                    "mse": float(mse_val),
                })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str,
                        default="/mnt/nas/jayden_code/Tablet-Detection-with-Object-Detection/data/yolo/train/images")
    parser.add_argument("--test_dir", type=str,
                        default="/mnt/nas/jayden_code/ai05-level1-project/test_images")
    parser.add_argument("--out_txt", type=str, default="duplicate_report.txt")
    parser.add_argument("--out_csv", type=str, default="duplicate_pairs.csv")
    parser.add_argument("--hash_threshold", type=int, default=5)
    parser.add_argument("--hist_threshold", type=float, default=0.95)
    parser.add_argument("--ssim_threshold", type=float, default=0.98)
    parser.add_argument("--mse_threshold", type=float, default=0.001)
    parser.add_argument("--hist_bins", type=int, default=16)
    args = parser.parse_args()

    print("[INFO] Parameters:")
    print(f"  train_dir      : {args.train_dir}")
    print(f"  test_dir       : {args.test_dir}")
    print(f"  hash_threshold : {args.hash_threshold}")
    print(f"  hist_threshold : {args.hist_threshold}")
    print(f"  ssim_threshold : {args.ssim_threshold}")
    print(f"  mse_threshold  : {args.mse_threshold}")
    print(f"  hist_bins      : {args.hist_bins}")
    print()

    results = find_duplicates(
        args.train_dir,
        args.test_dir,
        hash_threshold=args.hash_threshold,
        hist_threshold=args.hist_threshold,
        ssim_threshold=args.ssim_threshold,
        mse_threshold=args.mse_threshold,
        hist_bins=args.hist_bins,
    )

    # 결과 저장 (TXT)
    out_txt_path = Path(args.out_txt)
    with out_txt_path.open("w", encoding="utf-8") as f:
        f.write("=== Doubts of duplication (train ↔ test) ===\n")
        for item in results:
            f.write(
                f"[TRAIN] {item['train_img']}  <->  [TEST] {item['test_img']}\n"
                f"  hash_dist={item['hash_dist']}, "
                f"hist_sim={item['hist_sim']:.4f}, "
                f"ssim={item['ssim']:.4f}, "
                f"mse={item['mse']:.6f}\n\n"
            )

    # 결과 저장 (CSV)
    out_csv_path = Path(args.out_csv)
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "train_img", "train_path", "test_img", "test_path",
            "hash_dist", "hist_sim", "ssim", "mse"
        ])
        for item in results:
            writer.writerow([
                item["train_img"], item["train_path"],
                item["test_img"], item["test_path"],
                item["hash_dist"], f"{item['hist_sim']:.6f}",
                f"{item['ssim']:.6f}", f"{item['mse']:.8f}"
            ])

    print(f"\n[DONE] Duplicates found: {len(results)}")
    print(f"  TXT : {out_txt_path.resolve()}")
    print(f"  CSV : {out_csv_path.resolve()}")
    print("\nTip) CSV의 test_img 목록을 기준으로 test 세트에서 제거 후 재평가하세요.")


if __name__ == "__main__":
    main()
