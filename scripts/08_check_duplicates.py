# scripts/08_check_duplicates.py
from pathlib import Path
import hashlib
import csv
from tqdm import tqdm
from configs.paths import IMG_DIR, RUNS_DIR

OUT_CSV = RUNS_DIR / "duplicate_pairs.csv"  # 중복 결과 CSV 경로

def sha1(p: Path, chunk: int = 1 << 20) -> str:
    """바이트 SHA1 해시 계산"""
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def index_hashes(folder: Path):
    """파일명→해시 맵 생성"""
    idx = {}
    files = [p for p in folder.glob("*") if p.is_file()]
    for p in tqdm(files, desc=f"hash {folder.name}"):
        idx[p.name] = sha1(p)
    return idx

def main():
    train_dir = IMG_DIR / "train"
    val_dir   = IMG_DIR / "val"
    test_dir  = IMG_DIR / "test"

    train_h = index_hashes(train_dir)
    val_h   = index_hashes(val_dir)
    test_h  = index_hashes(test_dir)

    # 해시→이름 역맵
    rev_train = {}
    for k, v in train_h.items(): rev_train.setdefault(v, []).append(("train", k))
    rev_val = {}
    for k, v in val_h.items():   rev_val.setdefault(v, []).append(("val", k))
    rev_test = {}
    for k, v in test_h.items():  rev_test.setdefault(v, []).append(("test", k))

    dup_rows = []
    all_hashes = set(list(train_h.values()) + list(val_h.values()))
    inter = all_hashes.intersection(set(test_h.values()))

    for h in sorted(inter):
        for loc, fname in rev_test[h]:
            # test와 겹치는 train/val 목록 수집
            srcs = rev_train.get(h, []) + rev_val.get(h, [])
            for loc2, fname2 in srcs:
                dup_rows.append([h, f"{loc2}/{fname2}", f"{loc}/{fname}"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sha1", "train_or_val_file", "test_file"])
        w.writerows(dup_rows)

    print("pairs:", len(dup_rows))
    print("csv :", OUT_CSV)

if __name__ == "__main__":
    main()