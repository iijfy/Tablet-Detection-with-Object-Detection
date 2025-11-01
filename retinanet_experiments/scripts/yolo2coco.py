# retinanet_experiments/scripts/yolo2coco.py
import os, json, glob
from pathlib import Path

def load_classes(class_map_csv):
    # CSV: id,name  또는  name 만 있는 경우 모두 지원
    names = []
    with open(class_map_csv, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line or line.lower().startswith('id'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 1:
                names.append(parts[0])
            else:
                names.append(parts[1])
    return names

def yolo_to_coco(img_dir, lbl_dir, class_names, out_json):
    img_dir, lbl_dir = Path(img_dir), Path(lbl_dir)
    img_paths = sorted([p for p in img_dir.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}])

    images, annotations, categories = [], [], []
    for i, name in enumerate(class_names):
        categories.append({"id": i+1, "name": name})

    ann_id = 1
    img_id_map = {}
    for img_id, img_path in enumerate(img_paths, start=1):
        # 라벨 파일 추정 (이미지와 동일 경로 구조에서 확장자만 .txt)
        rel = img_path.relative_to(img_dir)
        lbl_path = lbl_dir / rel.with_suffix('.txt')

        # 이미지 크기는 모름 → COCO는 필수 아님(많은 툴에서 optional 처리), 있으면 더 좋지만 생략 가능
        images.append({
            "id": img_id,
            "file_name": str(rel).replace('\\','/'),
        })
        img_id_map[str(rel)] = img_id

        if not lbl_path.exists():
            continue

        with open(lbl_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:  # class cx cy w h (normalized)
                    continue
                cid = int(parts[0]) + 1  # COCO는 1부터, 배경은 사용 안함
                cx, cy, w, h = map(float, parts[1:])
                # xywh (pixel값 모르면 그대로 비정규화 없이 0~1 스케일로 두어도 대부분 로더가 허용)
                # 여기서는 img 크기를 모를 때의 안전책: 그대로 0~1 범위를 xywh로 기록
                # 추후 로더에서 실제 크기 곱이 필요하면 그때 보강 가능
                x = max(0.0, cx - w/2)
                y = max(0.0, cy - h/2)
                w = max(0.0, w)
                h = max(0.0, h)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cid,
                    "bbox": [x, y, w, h],
                    "area": w*h,
                    "iscrowd": 0,
                })
                ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(coco, f, ensure_ascii=False)
    print(f"[OK] saved: {out_json} | images={len(images)} anns={len(annotations)} cats={len(categories)}")

if __name__ == "__main__":
    # 프로젝트 기준 경로 (이전 단계 retinanet.yaml과 일치)
    IMG_ROOT = "/mnt/nas/jayden_code/ai05-level1-project/train_images"
    TRAIN_TXT_ROOT = "/mnt/nas/jayden_code/Tablet-Detection-with-Object-Detection/yolo/data_clean/train/labels"
    VAL_TXT_ROOT   = "/mnt/nas/jayden_code/Tablet-Detection-with-Object-Detection/yolo/data_clean/val/labels"
    CLASS_CSV      = "/mnt/nas/jayden_code/Tablet-Detection-with-Object-Detection/yolo/metadata/class_map.csv"

    OUT_TRAIN_JSON = "retinanet_experiments/data_coco/train.json"
    OUT_VAL_JSON   = "retinanet_experiments/data_coco/val.json"

    classes = load_classes(CLASS_CSV)
    print(f"classes: {len(classes)}")

    yolo_to_coco(IMG_ROOT, TRAIN_TXT_ROOT, classes, OUT_TRAIN_JSON)
    yolo_to_coco(IMG_ROOT, VAL_TXT_ROOT,   classes, OUT_VAL_JSON)