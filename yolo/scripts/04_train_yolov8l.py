# yolo/scripts/04_train_yolov8l.py
from ultralytics import YOLO
import os

def main():
    # === 경로 설정 ===
    project_dir = "yolo/runs"
    name = "yolov8l_baseline"  # 결과 저장 폴더 이름
    data_yaml = "yolo/data_clean/data.yaml"

    # === 모델 불러오기 ===
    model = YOLO("yolov8l.pt")  # Large 모델 (YOLOv8-Large)

    # === 학습 ===
    model.train(
        data=data_yaml,
        epochs=30,           # 에폭 수 (최초 테스트는 30회, 이후 늘려도 됨)
        imgsz=640,           # 이미지 크기
        batch=8,             # GPU 메모리에 따라 조절
        device=0,            # GPU 사용 (0번 GPU)
        project=project_dir,
        name=name,
        pretrained=True,     # COCO 사전학습 가중치 사용
        patience=20,         # 조기 종료 (validation loss 개선 없을 시)
        workers=2,           # dataloader 쓰레드
        exist_ok=True,       # 동일 이름 덮어쓰기 허용
    )

    print(f"✅ 학습 완료! 결과: {os.path.join(project_dir, name)}")

if __name__ == "__main__":
    main()