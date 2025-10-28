# yolo/scripts/05_make_submission.py
import os
import re
import sys
import yaml
import pandas as pd
from collections import Counter
from ultralytics import YOLO

# --------- 설정(필요시 경로만 바꾸면 됨) ----------
# 1) 테스트 이미지 디렉터리 (대회 제공)
TEST_DIR = "/mnt/nas/jayden_code/ai05-level1-project/test_images"

# 2) 학습에 썼던 data.yaml 경로 (names 읽을 때 필요)
#    아래 우선순위대로 탐색. 없으면 --data_yaml로 넘겨줘.
CANDIDATE_DATA_YAMLS = [
    "yolo/data_clean/data.yaml",
    "yolo/data/data.yaml",
    "yolo/data_clean.yaml",
]

# 3) 원본 라벨에서 class_name→orig_cat_id를 추출할 CSV
ANN_CSV = "yolo/annotations.csv"

# 4) 출력 제출 파일
OUT_CSV = "yolo/submission_yolov8l.csv"
# --------------------------------------------------

def find_data_yaml(cli_override: str | None) -> str:
    if cli_override:
        if os.path.isfile(cli_override):
            return cli_override
        raise FileNotFoundError(f"--data_yaml로 준 경로가 없어요: {cli_override}")
    for p in CANDIDATE_DATA_YAMLS:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "data.yaml을 찾을 수 없습니다. 아래 경로 중 하나에 두거나 --data_yaml로 지정해주세요:\n"
        + "\n".join(f" - {p}" for p in CANDIDATE_DATA_YAMLS)
    )

def load_yolo_names(data_yaml_path: str) -> list[str]:
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError(f"{data_yaml_path} 에서 names 리스트를 찾지 못했습니다.")
    # 문자열로 보정
    return [str(x) for x in names]

def build_name_to_catid_map(ann_csv_path: str) -> dict[str, int]:
    if not os.path.isfile(ann_csv_path):
        raise FileNotFoundError(
            f"원본 라벨 CSV({ann_csv_path})가 없습니다. 먼저 01_coco_to_csv.py를 실행해 주세요."
        )
    df = pd.read_csv(ann_csv_path)
    needed = {"class_name", "orig_cat_id"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"{ann_csv_path} 컬럼이 부족합니다. 필요: {sorted(needed)} / 현재: {list(df.columns)}"
        )
    # class_name마다 가장 빈도 높은 orig_cat_id를 대표값으로 사용
    mapping = {}
    for cls, g in df.groupby("class_name"):
        cnt = Counter(g["orig_cat_id"].tolist())
        rep = cnt.most_common(1)[0][0]
        mapping[str(cls)] = int(rep)
    return mapping

def ensure_bijective(yolo_names: list[str], name2cat: dict[str, int]) -> dict[int, int]:
    # yolo_id(0..N-1) -> category_id 로 변환
    missing = [n for n in yolo_names if n not in name2cat]
    if missing:
        raise ValueError(
            "다음 클래스 이름은 annotations_all.csv에서 orig_cat_id를 찾지 못했습니다:\n"
            + ", ".join(missing)
            + "\n- data.yaml의 names와 annotations_all.csv의 class_name이 정확히 일치하는지 확인하세요."
        )
    yolo2cat = {i: name2cat[n] for i, n in enumerate(yolo_names)}
    # category_id 중복 경고(대회가 허용하면 무시해도 되지만, 일단 확인)
    catids = list(yolo2cat.values())
    if len(catids) != len(set(catids)):
        # 중복 있으면 이유를 알려주기만 하고 계속 진행
        dup = [c for c, k in Counter(catids).items() if k > 1]
        print(f"[WARN] 서로 다른 YOLO 클래스가 같은 category_id로 매핑됩니다: {dup}")
    return yolo2cat

def parse_image_id(path: str) -> int:
    # 파일명에서 숫자만 뽑아 image_id로 사용 (예: "123.png" → 123)
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"\d+", stem)
    if not m:
        raise ValueError(f"이미지 파일명에서 image_id 숫자를 찾지 못했습니다: {path}")
    return int(m.group())

def main():
    if len(sys.argv) < 2:
        print("사용법: python yolo/scripts/05_make_submission.py <weights_path> [--data_yaml path/to/data.yaml]")
        sys.exit(1)

    weights = sys.argv[1]
    data_yaml_cli = None
    if "--data_yaml" in sys.argv:
        idx = sys.argv.index("--data_yaml")
        if idx + 1 >= len(sys.argv):
            raise ValueError("--data_yaml 다음에 경로를 적어주세요.")
        data_yaml_cli = sys.argv[idx + 1]

    if not os.path.isfile(weights):
        raise FileNotFoundError(f"가중치 파일이 없습니다: {weights}")

    data_yaml_path = find_data_yaml(data_yaml_cli)
    yolo_names = load_yolo_names(data_yaml_path)
    name2cat = build_name_to_catid_map(ANN_CSV)
    yolo2cat = ensure_bijective(yolo_names, name2cat)

    print(f"[INFO] Using weights : {weights}")
    print(f"[INFO] Using data.yaml : {data_yaml_path}")
    print(f"[INFO] Classes (YOLO idx -> category_id): {yolo2cat}")

    # 모델 로드 & 추론
    model = YOLO(weights)
    results = model.predict(
        source=TEST_DIR,
        imgsz=640,
        conf=0.001,   # 낮게 두고 점수 필터는 Kaggle 평가가 하므로 그대로 내보냄(필요시 조정)
        iou=0.7,
        device=0 if "CUDA_VISIBLE_DEVICES" in os.environ or os.path.isdir("/proc/driver/nvidia") else "cpu",
        save=False,
        verbose=False,
    )

    rows = []
    ann_id = 1
    for r in results:
        image_id = parse_image_id(r.path)
        # r.boxes.xywh, r.boxes.cls, r.boxes.conf
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xywh = r.boxes.xywh.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()

        for (x, y, w, h), c, s in zip(xywh, cls, conf):
            if c not in yolo2cat:
                raise ValueError(f"YOLO 클래스 인덱스 {c}에 대한 category_id 매핑이 없습니다.")
            category_id = yolo2cat[c]
            # 제출 스펙: 정수 bbox + 소수점 score
            rows.append([
                ann_id,
                image_id,
                int(category_id),
                int(round(float(x))),
                int(round(float(y))),
                int(round(float(w))),
                int(round(float(h))),
                float(s),
            ])
            ann_id += 1

    if not rows:
        raise RuntimeError("추론 결과가 비어 있습니다. weights, TEST_DIR, data.yaml을 확인하세요.")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df = pd.DataFrame(rows, columns=[
        "annotation_id","image_id","category_id","bbox_x","bbox_y","bbox_w","bbox_h","score"
    ])
    df.to_csv(OUT_CSV, index=False)
    print(f"✅ 제출 파일 저장 완료: {OUT_CSV} (총 {len(df)} rows)")
    print("   → Kaggle 요구 포맷과 동일합니다.")

if __name__ == "__main__":
    main()