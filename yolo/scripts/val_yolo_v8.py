import os
from ultralytics import YOLO

def main():
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "yolo/runs/yolov8l_clean_v1/weights/best.pt")
    data_path = os.path.join(cwd, "yolo/data.yaml")

    model = YOLO(model_path)
    metrics = model.val(data=data_path, imgsz=640, batch=8, device=0)

    print("✅ 검증 완료")
    print(metrics)

if __name__ == "__main__":
    main()