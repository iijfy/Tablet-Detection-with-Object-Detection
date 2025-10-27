# yolo/scripts/04_visualize_predictions.py
import os
from ultralytics import YOLO

def main():
    cwd = os.getcwd()
    model_path = os.path.join(cwd, "yolo/runs/yolov8l_clean_aug_v232/weights/best.pt")
    test_dir = os.path.join(cwd, "/mnt/nas/jayden_code/datasets/ai05-level1-project/test_images")
    save_dir = os.path.join(cwd, "yolo/runs/visualized_2")

    model = YOLO(model_path)
    model.predict(source=test_dir, imgsz=640, conf=0.25, save=True, project=save_dir, name="vis")

    print(f"✅ 예측 시각화 완료: {save_dir}")

if __name__ == "__main__":
    main()