# yolo/scripts/03_infer_and_make_submission.py
import os
import pandas as pd
from ultralytics import YOLO

def main():
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "yolo/runs/yolov8l_clean_aug_v232/weights/best.pt")
    test_dir = os.path.join(cwd, "/mnt/nas/jayden_code/ai05-level1-project/test_images")
    out_csv = os.path.join(cwd, "yolo/submission.csv")

    model = YOLO(model_path)
    results = model.predict(source=test_dir, imgsz=640, conf=0.25, device=0, save=False)

    preds = []
    ann_id = 0
    for r in results:
        image_id = int(os.path.splitext(os.path.basename(r.path))[0].split("_")[-1])
        for box in r.boxes:
            bbox = box.xywh[0].tolist()
            preds.append([
                ann_id,
                image_id,
                int(box.cls[0]),
                *[round(float(x), 2) for x in bbox],
                round(float(box.conf[0]), 2)
            ])
            ann_id += 1

    df = pd.DataFrame(preds, columns=["annotation_id", "image_id", "category_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"])
    df.to_csv(out_csv, index=False)
    print(f"✅ 제출파일 저장 완료: {out_csv}")

if __name__ == "__main__":
    main()