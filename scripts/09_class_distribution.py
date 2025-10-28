# scripts/09_class_distribution.py
from pathlib import Path
import json
import collections
import matplotlib.pyplot as plt
from configs.paths import ANN_DIR, RUNS_DIR

OUT_DIR = RUNS_DIR / "viz" / "class_dist"  # 출력 디렉터리

def load_coco_counts(coco_json: Path):
    """카테고리별 라벨 개수 집계"""
    coco = json.loads(coco_json.read_text(encoding="utf-8"))
    id2name = {c["id"]: c.get("name", str(c["id"])) for c in coco["categories"]}
    cnt = collections.Counter(a["category_id"] for a in coco["annotations"])
    names = [id2name[k] for k in cnt.keys()]
    counts = [cnt[k] for k in cnt.keys()]
    return names, counts

def plot_bar(names, counts, title, out_path: Path, top_k: int | None = 40):
    """상위 K 클래스 분포 그래프 저장"""
    pairs = list(zip(names, counts))
    pairs.sort(key=lambda x: x[1], reverse=True)
    if top_k:
        pairs = pairs[:top_k]
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    plt.figure(figsize=(12, 4))
    plt.bar(range(len(x)), y)
    plt.xticks(range(len(x)), x, rotation=90)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("saved:", out_path)

def main():
    train_json = ANN_DIR / "instances_train.json"
    val_json   = ANN_DIR / "instances_val.json"

    n_train, c_train = load_coco_counts(train_json)
    n_val,   c_val   = load_coco_counts(val_json)

    plot_bar(n_train, c_train, "Class Distribution (train)", OUT_DIR / "class_dist_train.png")
    plot_bar(n_val,   c_val,   "Class Distribution (val)",   OUT_DIR / "class_dist_val.png")

if __name__ == "__main__":
    main()