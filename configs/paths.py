# configs/paths.py
from pathlib import Path
import os

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 기본 원본 데이터 경로(없으면 환경변수로 대체)
_DEFAULTS = {
    "train_images": "/mnt/nas/jayden_code/ai05-level1-project/train_images",
    "test_images": "/mnt/nas/jayden_code/ai05-level1-project/test_images",
    "train_annotations": "/mnt/nas/jayden_code/ai05-level1-project/train_annotations",
}

# 환경변수 오버라이드
RAW_TRAIN_IMG = Path(os.getenv("AI05_TRAIN_IMG", _DEFAULTS["train_images"]))
RAW_TEST_IMG  = Path(os.getenv("AI05_TEST_IMG",  _DEFAULTS["test_images"]))
RAW_ANN_DIR   = Path(os.getenv("AI05_TRAIN_ANN", _DEFAULTS["train_annotations"]))

# 가공 데이터셋
DATASET_DIR = PROJECT_ROOT / "datasets" / "pills"
IMG_DIR     = DATASET_DIR / "images"
LABEL_DIR   = DATASET_DIR / "labels"
ANN_DIR     = DATASET_DIR / "annotations"

# 학습 설정/출력
YAML_PATH       = PROJECT_ROOT / "data" / "pills.yaml"
CLASS_MAP_PATH  = ANN_DIR / "class_map.json"
RUNS_DIR        = PROJECT_ROOT / "runs"
VIZ_DIR         = RUNS_DIR / "viz"
SUBMISSION_FILE = RUNS_DIR / "submission.csv"

# 디렉터리 생성
for p in [IMG_DIR, LABEL_DIR, ANN_DIR, VIZ_DIR, YAML_PATH.parent]:
    p.mkdir(parents=True, exist_ok=True)

# 원본 경로 유효성 검사
def assert_raw_paths():
    missing = [str(p) for p in [RAW_TRAIN_IMG, RAW_TEST_IMG, RAW_ANN_DIR] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing raw paths: " + ", ".join(missing) +
            "  set env AI05_TRAIN_IMG / AI05_TEST_IMG / AI05_TRAIN_ANN"
        )