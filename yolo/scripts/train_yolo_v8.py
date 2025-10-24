import os
from ultralytics import YOLO

def main():
    cwd = os.getcwd()
    data = os.path.join(cwd, "yolo/data.yaml")   # 데이터셋 경로
    save_root = "yolo/runs"
    run_name = "yolov8l_clean_v1"                # 실험 이름

    # Large 모델 로드
    model = YOLO("yolov8l.pt")                   # <-- L(large)

    results = model.train(
        data=data,
        epochs=20,                # 10으로 빠르게 확인 후 50~150으로 확장 추천
        imgsz=640,
        batch=4,                  # 8GB GPU 권장값
        workers=2,
        device=0,                 # CUDA:0
        project=save_root,
        name=run_name,
        pretrained=True,
        deterministic=True,       # 재현성
        seed=0,

        # 품질/속도 관련
        optimizer="auto",         # AdamW+lr 자동 탐색
        patience=50,              # 조기종료 여유
        close_mosaic=10,
        amp=True,

        # 로깅/체크포인트
        save=True,
        save_period=5,            # 5에폭마다 저장
        exist_ok=False,
        plots=True,
        val=True,                 # 매 epoch 검증
    )

    print("save_dir =", results.save_dir)

if __name__ == "__main__":
    main()