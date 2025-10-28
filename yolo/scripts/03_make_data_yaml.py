# yolo/scripts/03_make_data_yaml.py
import os
import yaml
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))  # yolo/
DATA_DIR = os.path.join(ROOT, "data_clean")
CSV_PATH = os.path.join(ROOT, "annotations.csv")
CLS_TXT = os.path.join(DATA_DIR, "classes.txt")
OUT_YAML = os.path.join(DATA_DIR, "data.yaml")

def load_class_names():
    # 1) classes.txt 우선
    if os.path.exists(CLS_TXT):
        with open(CLS_TXT, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        if names:
            print(f"✅ classes.txt 사용: {len(names)} classes")
            return names

    # 2) annotations_all.csv에서 추출
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV가 없습니다: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if "class_name" not in df.columns:
        raise ValueError("CSV에 'class_name' 컬럼이 없습니다.")

    names = sorted(df["class_name"].unique().tolist())
    print(f"✅ annotations_all.csv에서 클래스 추출: {len(names)} classes")
    # classes.txt도 남겨둠(재현성)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CLS_TXT, "w", encoding="utf-8") as f:
        for n in names:
            f.write(f"{n}\n")
    print(f"📝 classes.txt 저장 완료: {CLS_TXT}")
    return names

def main():
    names = load_class_names()
    nc = len(names)

    train_dir = os.path.join(DATA_DIR, "train", "images")
    val_dir   = os.path.join(DATA_DIR, "val", "images")

    # 경로 존재 체크
    for p in [train_dir, val_dir]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"경로가 없습니다: {p}")

    data = {
        "path": DATA_DIR,               # 선택: 상대경로 사용 시 편리
        "train": "train/images",
        "val": "val/images",
        "names": {i: n for i, n in enumerate(names)},
        "nc": nc,
    }

    with open(OUT_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"✅ data.yaml 생성 완료: {OUT_YAML}")
    print(f"   - nc: {nc}")
    print(f"   - train: {data['train']}")
    print(f"   - val:   {data['val']}")

if __name__ == "__main__":
    main()