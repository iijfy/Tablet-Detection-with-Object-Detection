#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_train_clean.py
- duplicate_pairs.csv 기준으로 중복(train↔test)된 train 이미지를 제외한
  'train_clean' 세트를 생성합니다.
- 이미지/라벨 동시 반영, data_clean.yaml 자동 생성, YOLO cache 제거까지 수행.

사용 예)
python tools/make_train_clean.py \
  --dup_csv duplicate_pairs.csv \
  --train_img_dir data/yolo/train/images \
  --train_lab_dir data/yolo/train/labels \
  --val_img_dir data/yolo/val/images \
  --project_root . \
  --out_subdir train_clean \
  --base_yaml yolo/data.yaml \
  --out_yaml yolo/data_clean.yaml \
  --mode copy     # 또는 move(원본에서 중복 파일을 옮겨 격리)
"""

import argparse
import os
import shutil
from pathlib import Path
import pandas as pd
import yaml
import sys
from tqdm import tqdm

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_duplicate_csv(csv_path: Path):
    """
    CSV 컬럼 호환:
    - mentor 코드 기준: train_img, train_path, test_img, test_path, hash_dist, hist_sim, ssim, mse
    - 혹시 컬럼명이 조금 달라도 'train_img'만 필요하므로 존재 확인
    """
    df = pd.read_csv(csv_path)
    # 유연성 확보: 소문자/공백 제거 비교
    columns = {c.lower().strip(): c for c in df.columns}
    if "train_img" in columns:
        col_train = columns["train_img"]
    else:
        # fallback: 가장 'train'과 'img'가 들어간 컬럼 찾기
        candidates = [c for c in df.columns if ("train" in c.lower() and "img" in c.lower())]
        if not candidates:
            print(f"[ERROR] CSV에서 'train_img' 컬럼을 찾을 수 없습니다. 컬럼들: {list(df.columns)}")
            sys.exit(1)
        col_train = candidates[0]
    dup_train_imgs = sorted(df[col_train].astype(str).unique().tolist())
    return dup_train_imgs, df

def copy_or_move(src: Path, dst: Path, mode: str):
    if mode == "copy":
        if src.exists():
            shutil.copy2(src, dst)
    elif mode == "move":
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    else:
        raise ValueError("mode must be 'copy' or 'move'")

def build_clean_split(
    train_img_dir: Path,
    train_lab_dir: Path,
    out_img_dir: Path,
    out_lab_dir: Path,
    dup_train_imgs: list,
    mode: str = "copy",
    removed_list_path: Path = None,
    quarantined_dir: Path = None,
):
    safe_mkdir(out_img_dir)
    safe_mkdir(out_lab_dir)

    removed = []
    kept = 0

    # 원본 train 전체 순회
    all_imgs = sorted([p for p in train_img_dir.iterdir() if p.is_file()])
    dup_set = set(dup_train_imgs)

    for img_path in tqdm(all_imgs, desc="Building train_clean"):
        fname = img_path.name
        stem = img_path.stem
        lab_path = train_lab_dir / f"{stem}.txt"

        if fname in dup_set:
            removed.append(fname)
            # 격리 모드(move)면 중복 파일들을 따로 옮겨두는 것도 가능
            if quarantined_dir:
                q_img = quarantined_dir / "images" / fname
                q_lab = quarantined_dir / "labels" / f"{stem}.txt"
                q_img.parent.mkdir(parents=True, exist_ok=True)
                q_lab.parent.mkdir(parents=True, exist_ok=True)
                if img_path.exists():
                    shutil.move(str(img_path), str(q_img))
                if lab_path.exists():
                    shutil.move(str(lab_path), str(q_lab))
            # copy 모드일 때는 원본을 건드리지 않음
            continue

        # 중복 아님 → clean으로 반영
        dst_img = out_img_dir / fname
        dst_lab = out_lab_dir / f"{stem}.txt"
        copy_or_move(img_path, dst_img, "copy")  # clean은 항상 copy로 구성
        if lab_path.exists():
            copy_or_move(lab_path, dst_lab, "copy")
        kept += 1

    # 제거 리스트 저장
    if removed_list_path:
        removed_list_path.parent.mkdir(parents=True, exist_ok=True)
        with removed_list_path.open("w", encoding="utf-8") as f:
            for r in removed:
                f.write(r + "\n")

    return kept, len(removed)

def write_data_clean_yaml(base_yaml: Path, out_yaml: Path, project_root: Path,
                          out_subdir: str, val_img_dir: Path):
    """
    base_yaml을 읽어 train 경로만 train_clean/images로 바꿔 out_yaml로 저장.
    base_yaml이 없으면 최소 필드로 새로 생성.
    """
    data = {}
    if base_yaml.exists():
        with base_yaml.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    # base yaml 구조 보존하면서 필요한 키 보정
    # 상대경로 사용 권장
    data["path"] = str(Path("data/yolo"))
    data["train"] = str(Path(f"data/yolo/{out_subdir}/images"))
    # 기존 val 유지, 없으면 인자로 받은 val 경로 반영
    if "val" not in data or not data["val"]:
        # 프로젝트 루트 기준 상대 경로로 변환
        rel_val = os.path.relpath(val_img_dir, project_root)
        data["val"] = rel_val.replace("\\", "/")

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

def clear_yolo_cache(root: Path):
    # data/yolo 아래 cache 삭제
    deleted = 0
    for p in root.rglob("*.cache"):
        try:
            p.unlink()
            deleted += 1
        except:
            pass
    return deleted

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dup_csv", required=True, help="duplicate_pairs.csv 경로")
    ap.add_argument("--train_img_dir", required=True, help="원본 train images 폴더")
    ap.add_argument("--train_lab_dir", required=True, help="원본 train labels 폴더")
    ap.add_argument("--val_img_dir", required=False, default="data/yolo/val/images", help="val images 폴더 (없으면 자동 보정)")
    ap.add_argument("--project_root", default=".", help="프로젝트 루트(상대경로 계산용)")
    ap.add_argument("--out_subdir", default="train_clean", help="data/yolo/<out_subdir>/ 하위에 images/labels 생성")
    ap.add_argument("--out_list", default="tools/removed_train_by_dup.txt", help="제외된 train 파일 리스트 저장 경로")
    ap.add_argument("--quarantine", default="data/yolo/train_dups", help="--mode move일 때 중복을 격리할 위치")
    ap.add_argument("--mode", choices=["copy","move"], default="copy",
                    help="copy: 원본 보존, move: 중복 파일을 quarantine 폴더로 이동")
    ap.add_argument("--base_yaml", default="yolo/data.yaml", help="기준 data.yaml 경로")
    ap.add_argument("--out_yaml", default="yolo/data_clean.yaml", help="생성될 data_clean.yaml 경로")
    return ap.parse_args()

def main():
    args = parse_args()

    dup_csv = Path(args.dup_csv).resolve()
    train_img_dir = Path(args.train_img_dir).resolve()
    train_lab_dir = Path(args.train_lab_dir).resolve()
    val_img_dir = Path(args.val_img_dir).resolve()
    project_root = Path(args.project_root).resolve()
    out_subdir = args.out_subdir

    out_img_dir = project_root / f"data/yolo/{out_subdir}/images"
    out_lab_dir = project_root / f"data/yolo/{out_subdir}/labels"
    removed_list_path = project_root / args.out_list
    quarantine_dir = project_root / args.quarantine if args.mode == "move" else None

    print("[INFO] Inputs:")
    print(f"  dup_csv     : {dup_csv}")
    print(f"  train_img   : {train_img_dir}")
    print(f"  train_lab   : {train_lab_dir}")
    print(f"  val_img_dir : {val_img_dir}")
    print(f"  out_subdir  : {out_subdir}  -> {out_img_dir}")
    print(f"  mode        : {args.mode}")

    dup_train_imgs, _df = read_duplicate_csv(dup_csv)
    print(f"[INFO] 중복(train↔test) 판정된 train 이미지 수: {len(dup_train_imgs)}")

    kept, removed = build_clean_split(
        train_img_dir, train_lab_dir,
        out_img_dir, out_lab_dir,
        dup_train_imgs,
        mode=args.mode,
        removed_list_path=removed_list_path,
        quarantined_dir=quarantine_dir,
    )

    print(f"[INFO] train_clean 구성 완료: kept={kept}, removed={removed}")
    print(f"[INFO] 제외 리스트 저장: {removed_list_path}")

    # data_clean.yaml 생성
    write_data_clean_yaml(
        base_yaml=Path(args.base_yaml),
        out_yaml=Path(args.out_yaml),
        project_root=project_root,
        out_subdir=out_subdir,
        val_img_dir=val_img_dir
    )
    print(f"[INFO] data_clean.yaml 생성: {args.out_yaml}")

    # YOLO cache 삭제
    deleted = clear_yolo_cache(project_root / "data/yolo")
    print(f"[INFO] YOLO cache 삭제: {deleted}개")

    print("\n[Next]")
    print(f"  1) 재학습:  yolo detect train model=yolov8l.pt data={args.out_yaml} name=yolov8l_clean_aug_v2 ...")
    print("  2) 학습 후 best.pt로 테스트 추론 및 제출 CSV 생성")

if __name__ == "__main__":
    main()