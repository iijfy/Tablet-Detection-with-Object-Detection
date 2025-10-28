# scripts/02_split_no_overlap.py
import os, json, hashlib, random, csv
from pathlib import Path
from tqdm import tqdm
import yaml
from configs.paths import (
    RAW_TRAIN_IMG, RAW_TEST_IMG, RAW_ANN_DIR,
    DATASET_DIR, IMG_DIR, ANN_DIR, YAML_PATH, CLASS_MAP_PATH, RUNS_DIR, assert_raw_paths
)

def sha1_hash(path: Path) -> str:
    buf = 1 << 20
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b: break
            h.update(b)
    return h.hexdigest()

def make_symlink_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        os.symlink(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)

def index_hashes(folder: Path):
    # 해시→파일명 목록
    m = {}
    files = [p for p in folder.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]]
    for p in tqdm(files, desc=f"hash {folder.name}"):
        h = sha1_hash(p)
        m.setdefault(h, []).append(p.name)
    return m

def main():
    assert_raw_paths()

    merged_json = RAW_ANN_DIR / "train_annotations_merged.json"
    assert merged_json.exists(), f"missing: {merged_json}"
    coco = json.loads(merged_json.read_text(encoding="utf-8"))

    images, annotations, categories = coco["images"], coco["annotations"], coco["categories"]

    train_hm = index_hashes(RAW_TRAIN_IMG)
    test_hm  = index_hashes(RAW_TEST_IMG)

    cross_dups = set(train_hm.keys()) & set(test_hm.keys())
    intra_dups = {h: v for h, v in train_hm.items() if len(v) > 1}

    train_keep = set()
    train_drop = []

    for h, names in train_hm.items():
        if h in cross_dups:
            for n in names:
                train_drop.append(("cross_dup", n))
            continue
        keep = names[0]
        train_keep.add(keep)
        for n in names[1:]:
            train_drop.append(("intra_dup", n))

    kept_images = [im for im in images if im["file_name"] in train_keep]
    dropped_ids = {im["id"] for im in images if im["file_name"] not in train_keep}
    kept_ann = [a for a in annotations if a["image_id"] not in dropped_ids]

    random.seed(42)
    random.shuffle(kept_images)
    n_train = int(len(kept_images) * 0.8)
    train_imgs = kept_images[:n_train]
    val_imgs   = kept_images[n_train:]

    def subset(subset_imgs):
        ids = {i["id"] for i in subset_imgs}
        anns = [a for a in kept_ann if a["image_id"] in ids]
        return {"images": subset_imgs, "annotations": anns, "categories": categories}

    train_coco = subset(train_imgs)
    val_coco   = subset(val_imgs)

    ANN_DIR.mkdir(parents=True, exist_ok=True)
    (ANN_DIR / "instances_train.json").write_text(json.dumps(train_coco, ensure_ascii=False, indent=2), encoding="utf-8")
    (ANN_DIR / "instances_val.json").write_text(json.dumps(val_coco,   ensure_ascii=False, indent=2), encoding="utf-8")

    class_map = {
        "id_to_name": {str(c["id"]): c["name"] for c in categories},
        "name_to_id": {c["name"]: c["id"] for c in categories},
        "id_to_index": {str(c["id"]): idx for idx, c in enumerate(sorted(categories, key=lambda x:x["id"]))},
    }
    CLASS_MAP_PATH.write_text(json.dumps(class_map, ensure_ascii=False, indent=2), encoding="utf-8")

    for subset, subset_imgs in [("train", train_imgs), ("val", val_imgs)]:
        for im in tqdm(subset_imgs, desc=f"link {subset}"):
            src = RAW_TRAIN_IMG / im["file_name"]
            dst = IMG_DIR / subset / im["file_name"]
            make_symlink_or_copy(src, dst)
    for p in tqdm(list(RAW_TEST_IMG.glob("*")), desc="link test"):
        make_symlink_or_copy(p, IMG_DIR / "test" / p.name)

    names = [c["name"] for c in sorted(categories, key=lambda x:x["id"])]
    ycfg = {
        "train": str((IMG_DIR / "train").resolve()),
        "val":   str((IMG_DIR / "val").resolve()),
        "test":  str((IMG_DIR / "test").resolve()),
        "nc":    len(names),
        "names": names,
    }
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(ycfg, f, allow_unicode=True, sort_keys=False)

    log_csv = RUNS_DIR / "dedup_log.csv"
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reason", "file_name"])
        w.writerows(train_drop)

    print("images raw:", len(images))
    print("train keep:", len(train_keep))
    print("drop intra dup:", sum(1 for r,_ in train_drop if r=='intra_dup'))
    print("drop cross dup:", sum(1 for r,_ in train_drop if r=='cross_dup'))
    print("train images:", len(train_imgs), "anns:", len(train_coco['annotations']))
    print("val   images:", len(val_imgs),   "anns:", len(val_coco['annotations']))
    print("yaml:", YAML_PATH)
    print("log :", log_csv)

if __name__ == "__main__":
    main()