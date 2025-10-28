# scripts/04_train_yolov8l.py
import os
from ultralytics import YOLO
from configs.paths import YAML_PATH, RUNS_DIR

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model = YOLO("yolov8l.pt")

    model.train(
        data=str(YAML_PATH),
        epochs=60,
        patience=20,
        imgsz=896,
        batch=2,
        workers=4,
        device=0,
        project=str(RUNS_DIR),
        name="yolov8l_tuned",
        pretrained=True,
        seed=42,
        deterministic=True,

        optimizer="AdamW",
        lr0=0.0007,
        cos_lr=True,

        degrees=0.0,
        flipud=0.0,
        fliplr=0.1,
        mosaic=0.25,
        scale=0.25,
        translate=0.04,
        hsv_h=0.0, hsv_s=0.2, hsv_v=0.2,
        close_mosaic=20,

        box=6.0, cls=0.4, dfl=1.5,
        rect=True,
        plots=True,
        exist_ok=True
    )

    print("run dir:", RUNS_DIR / "yolov8l_tuned")

if __name__ == "__main__":
    main()