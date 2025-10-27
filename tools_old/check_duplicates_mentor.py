import os
from PIL import Image
import imagehash
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


def load_image_rgb(path, target_size=(256, 256)):

    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(target_size)
        arr = np.asarray(img, dtype=np.float32) / 255.0

    return arr


def compute_hash_and_hist(path, hist_bins=16):

    with Image.open(path) as img:
        img_rgb = img.convert("RGB")
        # perceptual hash (밝기 기반 구조 특성)
        h = imagehash.phash(img_rgb)

        # color histogram
        arr = np.asarray(img_rgb, dtype=np.uint8)
        hist_feats = []
        for c in range(3):  # R,G,B
            hist, _ = np.histogram(
                arr[..., c],
                bins=hist_bins,
                range=(0, 256),
                density=True,
            )
            hist_feats.append(hist)
        hist_vec = np.concatenate(hist_feats)  # shape (hist_bins*3,)

    return h, hist_vec


def cosine_similarity(a, b, eps=1e-8):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return dot / (na * nb + eps)


def mse(a, b):
    diff = a - b
    return float(np.mean(diff * diff))


def compare_pair(img_a_path, img_b_path,
                 size_for_diff=(256, 256)):

    A = load_image_rgb(img_a_path, target_size=size_for_diff)
    B = load_image_rgb(img_b_path, target_size=size_for_diff)

    # skimage.metrics.ssim 은 channel_axis 인자를 사용해 컬러 SSIM 가능
    ssim_val = ssim(A, B, channel_axis=2, data_range=1.0)
    mse_val = mse(A, B)
    return ssim_val, mse_val


def build_index(folder):

    index = []

    for fname in tqdm(os.listdir(folder), desc=f"Indexing {folder}"):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            h, hist_vec = compute_hash_and_hist(fpath)
            index.append({
                "name": fname,
                "path": fpath,
                "hash": h,
                "hist": hist_vec,
            })
        except Exception as e:
            print(f"[WARN] {fname}: {e}")
    return index


def find_duplicates(train_dir, test_dir,
                    hash_threshold=5,
                    hist_threshold=0.95,
                    ssim_threshold=0.98,
                    mse_threshold=0.001):

    train_index = build_index(train_dir)
    test_index = build_index(test_dir)

    results = []

    # 비교
    for t_item in tqdm(test_index, desc="Cross-check train vs test"):
        for tr_item in train_index:
            # 1차 필터: 구조
            hash_dist = abs(t_item["hash"] - tr_item["hash"])
            if hash_dist > hash_threshold:
                continue

            # 1차 필터: 색 분포 유사성
            hist_sim = cosine_similarity(t_item["hist"], tr_item["hist"])
            if hist_sim < hist_threshold:
                continue

            # 2차 정밀 비교
            ssim_val, mse_val = compare_pair(t_item["path"], tr_item["path"])

            if ssim_val >= ssim_threshold and mse_val <= mse_threshold:
                results.append({
                    "train_img": tr_item["name"],
                    "test_img": t_item["name"],
                    "hash_dist": hash_dist,
                    "hist_sim": hist_sim,
                    "ssim": ssim_val,
                    "mse": mse_val,
                })

    return results


if __name__ == "__main__":
    train_dir = "/mnt/nas/jayden_code/Tablet-Detection-with-Object-Detection/data/yolo/train/images"
    test_dir = "/mnt/nas/jayden_code/ai05-level1-project/test_images"

    dup_pairs = find_duplicates(
        train_dir,
        test_dir,
        hash_threshold=5,       # 구조적으로 거의 동일
        hist_threshold=0.95,    # 색 분포 거의 동일
        ssim_threshold=0.98,    # 실제 픽셀 구조+색이 사실상 동일
        mse_threshold=0.001     # 픽셀 차이가 거의 없는 수준
    )

    print("\n=== Doubts Data of duplication (train ↔ test) ===")
    
    for item in dup_pairs:
        print(
            f"[TRAIN] {item['train_img']}  <->  [TEST] {item['test_img']}\n"
            f"  hash_dist={item['hash_dist']}, "
            f"hist_sim={item['hist_sim']:.4f}, "
            f"ssim={item['ssim']:.4f}, "
            f"mse={item['mse']:.6f}\n"
        )