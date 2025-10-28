# yolo/scripts/04_visualize_predictions.py
import os
from pathlib import Path
import cv2
from ultralytics import YOLO


def main():
    # 현재 작업 디렉토리
    cwd = os.getcwd()

    # ✅ 경로 설정 (각자 환경에 맞게 수정)
    model_path = os.path.join(cwd, "yolo/runs/yolov8l_clean_aug_v232/weights/best.pt")
    test_dir   = "/mnt/nas/jayden_code/ai05-level1-project/test_images"   # 실제 테스트 이미지 경로
    save_dir   = os.path.join(cwd, "yolo/runs/visualized_aug_v232")

    # 폴더 생성
    os.makedirs(save_dir, exist_ok=True)

    # ✅ 모델 불러오기
    model = YOLO(model_path)

    # ✅ 예측 실행 (이미지별로 시각화 저장)
    results = model.predict(
        source=test_dir,
        imgsz=640,
        conf=0.25,
        save=False,          # YOLO 기본 저장 비활성화
        device=0
    )

    print(f"\n✅ 총 {len(results)}개 이미지 예측 완료!")

    # ✅ 클래스별로 폴더를 나누어 저장
    for r in results:
        im0 = r.plot()  # YOLO가 그린 결과 이미지
        image_name = Path(r.path).stem

        # 결과에 포함된 객체(box)별로 클래스 이름 확인
        if len(r.boxes) == 0:
            cls_name = "no_detection"
            save_path = os.path.join(save_dir, cls_name)
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, f"{image_name}.jpg"), im0)
            continue

        # 여러 클래스가 있을 경우 각각에 대해 저장
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            cls_dir = os.path.join(save_dir, cls_name)
            os.makedirs(cls_dir, exist_ok=True)
            cv2.imwrite(os.path.join(cls_dir, f"{image_name}.jpg"), im0)

    print(f"🎨 예측 이미지가 클래스별로 저장되었습니다 → {save_dir}")


if __name__ == "__main__":
    main()