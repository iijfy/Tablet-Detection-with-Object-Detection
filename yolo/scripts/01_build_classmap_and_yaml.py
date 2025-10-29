# -*- coding: utf-8 -*-
import os, glob, json, csv

ANN_JSON_DIR = "/mnt/nas/jayden_code/ai05-level1-project/train_annotations"
YOLO_DIR      = os.path.dirname(os.path.dirname(__file__))     # .../yolo
META_DIR      = os.path.join(YOLO_DIR, "metadata")
CLASSMAP_CSV  = os.path.join(META_DIR, "class_map.csv")
DATA_YAML     = os.path.join(YOLO_DIR, "data.yaml")
DATA_ROOT     = os.path.join(YOLO_DIR, "data_clean")

def collect_categories():
    id2name = {}
    for jp in glob.glob(os.path.join(ANN_JSON_DIR, "**/*.json"), recursive=True):
        try:
            d = json.load(open(jp, "r", encoding="utf-8"))
            for c in d.get("categories", []):
                cid = int(c["id"])
                cname = str(c.get("name", f"cls_{cid}"))
                if cid not in id2name:
                    id2name[cid] = cname
        except Exception:
            pass
    return id2name

def main():
    os.makedirs(META_DIR, exist_ok=True)
    id2name = collect_categories()
    assert id2name, "JSON에서 categories를 찾지 못했습니다."

    # yolo_id는 orig_cat_id를 오름차순 정렬한 인덱스
    sorted_ids = sorted(id2name.keys())
    with open(CLASSMAP_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["orig_cat_id", "yolo_id", "class_name"])
        for i, cid in enumerate(sorted_ids):
            wr.writerow([cid, i, id2name[cid]])
    print(f"✅ class_map.csv 저장: {CLASSMAP_CSV} (classes={len(sorted_ids)})")

    names = [id2name[cid] for cid in sorted_ids]

    # 따옴표 조립을 분리해서 문자열 먼저 만든다
    names_str = ", ".join([f"'{n}'" for n in names])

    yaml_lines = [
        f"path: {DATA_ROOT}",
        "train: train/images",
        "val: val/images",
        f"names: [{names_str}]",
    ]

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(f"✅ data.yaml 저장: {DATA_YAML} (names={len(names)})")

if __name__ == "__main__":
    main()