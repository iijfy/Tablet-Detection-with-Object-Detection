# yolo/scripts/01_make_annotations_csv.py
import os, json, csv, glob

# === 원본 데이터 (NAS) ===
ANN_DIR = "/mnt/nas/jayden_code/ai05-level1-project/train_annotations"

# === 출력 CSV (프로젝트 내부) ===
OUT_CSV = "yolo/annotations.csv"

def parse_one(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # image_id -> (file_name, width, height)
    image_map = {im["id"]: (im["file_name"], im["width"], im["height"])
                 for im in data.get("images", [])}

    # category_id -> name
    cat_map = {c["id"]: str(c.get("name", f"class_{c['id']}"))
               for c in data.get("categories", [])}

    rows = []
    for ann in data.get("annotations", []):
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        img_id = ann["image_id"]
        if img_id not in image_map:
            continue
        fname, W, H = image_map[img_id]
        cname = cat_map.get(ann["category_id"], "UNK_CLASS")

        # CSV 스키마(고정): 학습/스플릿/시각화에서 그대로 재사용
        rows.append([fname, cname, x, y, w, h, ann["category_id"], W, H])
    return rows

def main():
    files = sorted(glob.glob(os.path.join(ANN_DIR, "*.json")))
    assert files, f"JSON 파일을 찾지 못했습니다: {ANN_DIR}"

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    count = 0
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["filename","class_name","x","y","w","h","orig_cat_id","img_w","img_h"])
        for jp in files:
            rows = parse_one(jp)
            wr.writerows(rows)
            count += len(rows)
    print(f"✅ CSV 저장 완료: {OUT_CSV} (총 {count}개 어노테이션)")

if __name__ == "__main__":
    main()