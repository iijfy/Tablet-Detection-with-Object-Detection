# COCO(JSON) -> CSV (bbox + class + img_w/h까지)
import os, json, glob, csv

ANN_DIR = "/mnt/nas/jayde_code/datasets/ai05-level1-project/train_annotations"
OUT_CSV = "yolo/annotations_all.csv"

def parse_one(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    # id -> (file_name, width, height)
    imap = {im["id"]: (im["file_name"], im["width"], im["height"])
            for im in d.get("images", [])}

    # category id -> name
    cmap = {c["id"]: str(c.get("name", f"class_{c['id']}"))
            for c in d.get("categories", [])}

    rows = []
    for a in d.get("annotations", []):
        bbox = a.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        imgid = a["image_id"]
        if imgid not in imap:
            continue
        fname, W, H = imap[imgid]
        cname = cmap.get(a["category_id"], "UNK_CLASS")
        rows.append([fname, cname, x, y, w, h, a["category_id"], W, H])
    return rows

def main():
    files = glob.glob(os.path.join(ANN_DIR, "**/*.json"), recursive=True)
    assert files, f"JSON을 찾지 못함: {ANN_DIR}"

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["filename","class_name","x","y","w","h","orig_cat_id","img_w","img_h"])
        for jp in files:
            for r in parse_one(jp):
                wr.writerow(r)
    print(f"✅ CSV 저장: {OUT_CSV} (총 {len(files)}개 JSON 스캔)")

if __name__ == "__main__":
    main()