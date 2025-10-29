# -*- coding: utf-8 -*-
# YOLOv8l 학습 스크립트
from ultralytics import YOLO

def main():
    model = YOLO("yolov8l.pt")  # pretrained large model

    model.train(
        data="yolo/data.yaml",       # 이전 단계에서 생성된 data.yaml
        epochs=50,                   # 학습 epoch 수
        batch=8,
        imgsz=768,
        device=0,                    # GPU 0번 사용
        project="yolo/runs",
        name="yolov8l_baseline",
        workers=4,
        patience=30,                 # 성능 개선 없으면 조기종료
        seed=42,
        deterministic=True,
        pretrained=True,
        cos_lr=True,                 # learning rate cosine schedule
        warmup_epochs=3,
        label_smoothing=0.05,        # label smoothing
        box=7.5, cls=0.5, dfl=1.5,   # loss balance
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # 색상 증강
        translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.0, erasing=0.4,
        close_mosaic=10,
    )

if __name__ == "__main__":
    main()