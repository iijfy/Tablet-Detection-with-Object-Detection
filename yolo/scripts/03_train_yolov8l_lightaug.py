# -*- coding: utf-8 -*-
"""YOLOv8-L 학습 (Light Aug 전용버전) 😊
- 기존 스크립트를 건드리지 않고 '새 파일'로 학습해요
- 핵심: 조명/색/약한 기하 증강만 적용, mosaic/mixup은 끔
- 결과 경로는 project/name 으로 분리되어 기존 실험과 충돌하지 않아요
"""

from ultralytics import YOLO
import os
from datetime import datetime
import getpass
import subprocess

# ============ 사용자 환경 ============
DATA_YAML = "yolo/data.yaml"  # yolo/data.yaml (data_clean 기준으로 작성된 yaml)
BASE_WEIGHTS = "yolov8l.pt"  # 또는 이전 best.pt 로 warm-start 가능
IMG_SIZE = 640            # 640~768 권장
EPOCHS = 50
BATCH = 8
DEVICE = 0               # 여러 GPU면 "0,1" 형식 가능

# ============ 실험 메타(폴더/이름) ============
def get_git_branch():
    try:
        out = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "nogit"

BRANCH = get_git_branch()
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M")
USER = getpass.getuser()

# Ultralytics는 project/name 으로 출력 경로를 나눌 수 있어요
PROJECT = os.path.join("runs", "yolov8l_lightaug")          # 상위 폴더
NAME = f"lightaug_{BRANCH}_{USER}_{RUN_ID}"                 # 하위 폴더(런 이름)

def main():
    # 1) 모델 로드
    model = YOLO(BASE_WEIGHTS)  # "yolov8l.pt" 또는 이전 학습 가중치 경로

    # 2) 학습 (Light Aug 설정)
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        workers=8,

        # ===== Light Aug (광학 + 약한 기하) =====
        hsv_h=0.015,     # 🎨 Hue 작은 변화
        hsv_s=0.7,       # Saturation
        hsv_v=0.4,       # Value(밝기)
        fliplr=0.5,      # ↔ 좌우 반전 50%
        flipud=0.0,      # ↕ 상하 반전 X
        degrees=7.0,     # 🔄 회전 ±7도
        translate=0.05,  # 📦 평행이동 5%
        scale=0.15,      # 🔍 스케일 ±15%
        shear=0.0,       # 기울이기 X
        perspective=0.0, # 원근 X
        mosaic=0.0,      # ❌ 모자이크 끔
        mixup=0.0,       # ❌ 믹스업 끔
        copy_paste=0.0,  # ❌

        # ===== 최적화/학습 안정 =====
        optimizer="SGD",       # 또는 "AdamW"
        lr0=0.01, lrf=0.1,
        momentum=0.937,
        weight_decay=0.0005,
        patience=20,           # EarlyStopping

        # ===== 출력 경로 관리 =====
        project=PROJECT,       # runs/yolov8l_baseline
        name=NAME,             # lightaug_<branch>_<user>_<time>
        exist_ok=False,        # 같은 이름 있으면 에러(덮어쓰기 방지)
        pretrained=True
    )

    print(f"[INFO] 결과 폴더: {os.path.join(PROJECT, NAME)}")
    print("[TIP] best.pt, results.csv, confusion_matrix.png 등을 확인하세요!")

if __name__ == "__main__":
    main()