# yolo/scripts/01_coco_to_csv.py
# COCO 어노테이션들을 평탄화하여 하나의 CSV로 합치기
import os, json, glob, csv

ANN_DIR = "/home/jayden86/datasets/ai05-level1-project/train_annotations"
OUT_CSV = "yolo/annotations_all.csv"

def parse_one(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    id2name = {c["id"]: str(c.get("name", f"class_{c['id']}")) for c in d.get("categories", [])}
    id2file = {im["id"]: im["file_name"] for im in d.get("images", [])}

    rows = []
    for a in d.get("annotations", []):
        bbox = a.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        rows.append([
            id2file.get(a["image_id"], "UNK.png"),
            id2name.get(a["category_id"], "UNK_CLASS"),
            x, y, w, h,
            a["category_id"]  # 원본 카테고리 id (재매핑용)
        ])
    return rows

def main():
    files = glob.glob(os.path.join(ANN_DIR, "**/*.json"), recursive=True)
    assert files, f"JSON을 찾지 못함: {ANN_DIR}"
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["filename","class_name","x_min","y_min","width","height","orig_cat_id"])
        for jp in files:
            for r in parse_one(jp):
                wr.writerow(r)
    print(f"✅ CSV 저장: {OUT_CSV}")

if __name__ == "__main__":
    main()