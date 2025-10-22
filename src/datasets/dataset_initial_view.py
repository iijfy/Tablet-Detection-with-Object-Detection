# src/data.py
import os
import shutil
import numpy as np
from collections import Counter


def copy_all_json_files(source_dir: str, target_dir: str):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    os.makedirs(target_dir, exist_ok=True)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}\n")

    count = 0
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".json"):
                src = os.path.join(root, file)
                dst = os.path.join(target_dir, file)
                if os.path.exists(dst):
                    base, ext = os.path.splitext(file)
                    folder = os.path.basename(root)
                    dst = os.path.join(target_dir, f"{base}_{folder}{ext}")
                shutil.copy2(src, dst)
                count += 1
                print(f"Copied: {src} → {dst}")

    print(f"\n 총 {count}개의 JSON 파일이 복사되었습니다.")

import os
import json
import pandas as pd

def flatten_json_section(section_name, items):
    """JSON 리스트를 DataFrame으로 평면화하고 prefix 추가"""
    if not items:
        return pd.DataFrame()
    df = pd.json_normalize(items)
    df.columns = [f"{section_name}_{col}" for col in df.columns]
    return df

def load_json_to_dataframe(folder_path: str, df_name: str = "merged_df"):
    """
    지정된 폴더의 모든 JSON 파일을 읽어 평면화 후 단일 DataFrame으로 병합하여 반환.
    
    Args:
        folder_path (str): JSON 파일들이 들어 있는 디렉터리 경로
        df_name (str): 생성할 DataFrame 이름 (옵션)
    
    Returns:
        pd.DataFrame: 모든 JSON 데이터를 병합한 DataFrame
    """
    all_dfs = []

    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"JSON 디코딩 실패: {file_path} — {e}")
                    continue

            # 각 JSON 내 section별 평면화
            df_images = flatten_json_section("images", data.get("images", []))
            df_annotations = flatten_json_section("annotations", data.get("annotations", []))
            df_categories = flatten_json_section("categories", data.get("categories", []))

            # 병합 (id 매칭 기준)
            if not df_annotations.empty:
                df_merged = (
                    df_annotations
                    .merge(df_images, left_on="annotations_image_id", right_on="images_id", how="left")
                    .merge(df_categories, left_on="annotations_category_id", right_on="categories_id", how="left")
                )
            else:
                # annotations가 없을 경우 단순 결합
                df_merged = pd.concat([df_images, df_categories], axis=1)

            df_merged["source_file"] = file  # JSON 파일 이름 추적용
            all_dfs.append(df_merged)

    if not all_dfs:
        print("폴더 내에서 JSON 파일을 찾을 수 없습니다.")
        return pd.DataFrame()

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # 전역 변수명으로 지정 (선택적)
    globals()[df_name] = merged_df
    print(f"{df_name} 생성 완료 (총 {len(merged_df)}행)")

    return merged_df

import os
import pandas as pd

def load_image_filenames_to_df(folder_path: str, df_name: str = "df_train_images"):
    """
    지정된 폴더 내의 파일명을 file_name 컬럼으로 하는 DataFrame을 생성합니다.
    
    Args:
        folder_path (str): 이미지 파일들이 들어 있는 폴더 경로
        df_name (str): 생성할 DataFrame 이름 (옵션)
    
    Returns:
        pd.DataFrame: file_name 컬럼을 가진 DataFrame
    """
    # 폴더 존재 확인
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"❌ 폴더를 찾을 수 없습니다: {folder_path}")

    # 파일명 리스트 생성
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # DataFrame 생성
    df = pd.DataFrame({"file_name": file_names})
    
    # 전역 변수 등록 (선택적)
    globals()[df_name] = df
    print(f"{df_name} 생성 완료 (총 {len(df)}개 파일)")
    
    return df

def compare_filenames(train_anno_df: pd.DataFrame, train_images_df: pd.DataFrame):
    """
    annotations와 images의 파일명을 비교하여 누락/불일치/공통 항목을 출력합니다.

    Args:
        train_anno_df (pd.DataFrame): JSON에서 추출한 annotation DataFrame (컬럼명: images_file_name)
        train_images_df (pd.DataFrame): 이미지 파일 리스트 DataFrame (컬럼명: file_name)

    Returns:
        dict: 각 비교 결과를 담은 딕셔너리
    """
    anno_files = set(train_anno_df["images_file_name"].dropna().unique())
    image_files = set(train_images_df["file_name"].dropna().unique())

    # 차집합 및 교집합 계산
    missing_in_images = sorted(list(anno_files - image_files))      # annotation에는 있으나 실제 폴더엔 없는 파일
    missing_in_annotations = sorted(list(image_files - anno_files)) # 폴더엔 있으나 annotation엔 없는 파일
    common_files = sorted(list(anno_files & image_files))           # 양쪽 모두 있는 파일

    # 결과 출력
    print("비교 결과:")
    print(f"annotation에는 있으나 image 폴더에는 없는 파일 수: {len(missing_in_images)}")
    print(f"image 폴더에는 있으나 annotation에는 없는 파일 수: {len(missing_in_annotations)}")
    print(f"양쪽 모두 존재하는 공통 파일 수: {len(common_files)}\n")

    # annotation엔 있으나 이미지엔 없는 파일
    if missing_in_images:
        print("annotation에는 있으나 이미지에 없는 파일:")
        for f in missing_in_images[:3]:
            print("  -", f)
        if len(missing_in_images) > 3:
            print(f"  ... (총 {len(missing_in_images)}개)\n")

    # 이미지엔 있으나 annotation엔 없는 파일
    if missing_in_annotations:
        print("이미지에는 있으나 annotation에 없는 파일:")
        for f in missing_in_annotations[:3]:
            print("  -", f)
        if len(missing_in_annotations) > 3:
            print(f"  ... (총 {len(missing_in_annotations)}개)\n")

    # 양쪽 모두 존재하는 파일
    if common_files:
        print("양쪽 모두 존재하는 파일 (일부 표시):")
        for f in common_files[:3]:
            print("  -", f)
        if len(common_files) > 3:
            print(f"  ... (총 {len(common_files)}개)\n")

    return {
        "missing_in_images": missing_in_images,
        "missing_in_annotations": missing_in_annotations,
        "common_files": common_files
    }


def expand_bbox_columns(df: pd.DataFrame):
    """
    annotations_bbox가 문자열("[x, y, w, h]") 또는 리스트 형태일 때
    안전하게 [bbox_x, bbox_y, bbox_w, bbox_h] 컬럼으로 확장.
    """

    def parse_bbox(bbox):
        # ① 결측값 또는 빈 리스트 처리
        if bbox is None or bbox == [] or bbox == "":
            return [None, None, None, None]

        # ② 문자열인 경우 (예: "[100, 200, 50, 60]")
        if isinstance(bbox, str):
            try:
                bbox = ast.literal_eval(bbox)  # 문자열 → 리스트
            except Exception:
                return [None, None, None, None]

        # ③ 중첩 리스트 ([[100, 200, 50, 60]]) 처리
        if isinstance(bbox, list) and len(bbox) == 1 and isinstance(bbox[0], list):
            bbox = bbox[0]

        # ④ 길이가 4인 리스트인지 검사
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
            return bbox
        else:
            return [None, None, None, None]

    # bbox 파싱
    parsed_bboxes = df["annotations_bbox"].apply(parse_bbox)

    # 개별 컬럼 분해
    bbox_df = pd.DataFrame(parsed_bboxes.tolist(), columns=["bbox_x", "bbox_y", "bbox_w", "bbox_h"])

    # 원본 DataFrame에서 annotations_bbox 제거 후 합치기
    df_expanded = pd.concat([df.drop(columns=["annotations_bbox"]), bbox_df], axis=1)

    return df_expanded


def validate_bbox(df: pd.DataFrame):
    """
    bbox_x, bbox_y, bbox_w, bbox_h 컬럼이
    1️⃣ 모두 존재하고
    2️⃣ NaN 없이
    3️⃣ 정수(int) 타입이며
    4️⃣ w, h > 0
    인지 검증합니다.

    Returns:
        dict: 검증 결과 요약 및 오류 DataFrame 포함
    """
    required_cols = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    
    print("[BBox Validation] 시작")

    # 1. 필수 컬럼 존재 확인
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"필수 bbox 컬럼이 없습니다: {missing_cols}")
        return {"missing_columns": missing_cols, "invalid_rows": pd.DataFrame()}

    # 2. 각 행별 유효성 검사
    invalid_rows = df[
        df[required_cols].isnull().any(axis=1) |  # 하나라도 NaN인 경우
        df[required_cols].applymap(lambda x: not isinstance(x, (int, np.integer))).any(axis=1) |  # 정수가 아닌 경우
        (df["bbox_w"] <= 0) | (df["bbox_h"] <= 0)  # 너비/높이 0 이하
    ]

    # 결과 출력
    total = len(df)
    invalid_count = len(invalid_rows)
    valid_count = total - invalid_count

    print(f"총 {total}개 중 정상 {valid_count}개, 오류 {invalid_count}개")

    if invalid_count > 0:
        print("잘못된 BBox 예시:")
        print(invalid_rows.head(10)[["annotations_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]])

    print("검증 완료\n")
    return {
        "total": total,
        "valid": valid_count,
        "invalid": invalid_count,
        "invalid_rows": invalid_rows
    }


import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches

def visualize_bbox_grid_from_image_df(
    images_df,
    anno_df,
    image_dir="./datafile/train_images",
    start=0,
    end=8,
    ncols=4,
    save_dir="./outputs",
    show=True
):
    """
    train_images_df 기준으로 이미지를 순회하며,
    train_anno_df에서 bbox 정보를 찾아 시각화합니다.

    Args:
        images_df (pd.DataFrame): 이미지 파일 목록 (file_name 컬럼 필수)
        anno_df (pd.DataFrame): bbox_x, bbox_y, bbox_w, bbox_h, categories_name, images_file_name 포함된 DF
        image_dir (str): 이미지 폴더 경로
        start (int): 시작 인덱스 (images_df 기준)
        end (int): 종료 인덱스 (images_df 기준, 미포함)
        ncols (int): 한 행당 이미지 개수
        save_dir (str): 이미지 저장 폴더
        show (bool): Notebook/GUI 환경에서는 True → plt.show(), 터미널에서는 False → 파일로 저장
    """
    # images_df에서 해당 구간 이미지 선택
    subset = images_df.iloc[start:end]
    unique_images = subset["file_name"].unique()
    n_images = len(unique_images)
    nrows = (n_images + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
    axes = axes.flatten() if n_images > 1 else [axes]
    colors = plt.cm.tab10.colors

    for i, image_name in enumerate(unique_images):
        image_path = os.path.join(image_dir, image_name)
        ax = axes[i]

        if not os.path.exists(image_path):
            ax.set_title(f"❌ Not Found:\n{image_name}", fontsize=8)
            ax.axis("off")
            continue

        # 이미지 로드
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        ax.axis("off")

        # anno_df에서 해당 이미지의 bbox 검색
        img_rows = anno_df[anno_df["images_file_name"] == image_name]

        # bbox 그리기
        for j, (_, item) in enumerate(img_rows.iterrows()):
            x, y, w, h = item["bbox_x"], item["bbox_y"], item["bbox_w"], item["bbox_h"]
            name = item["categories_name"]
            color = colors[j % len(colors)]

            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(
                x, y - 5, name,
                fontsize=9, color="white", backgroundcolor=color, fontweight="bold"
            )

        ax.set_title(f"{image_name}", fontsize=8)

    # 남은 빈칸 숨기기
    for k in range(i + 1, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()

    # 환경에 따라 show/save 자동 분기
    if not show or not sys.stdout.isatty():
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"bbox_grid_{start}_{end}.png")
        plt.savefig(output_path, dpi=150)
        print(f"시각화 결과 저장 완료: {output_path}")
        plt.close(fig)
    else:
        plt.show()


def analyze_image_folder(folder_path: str):
    image_info = []
    file_types = Counter()

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        mode = img.mode  # e.g., 'RGB', 'L', 'RGBA'
                        channels = len(img.getbands())
                        fmt = img.format
                        image_info.append((width, height, channels))
                        file_types[fmt] += 1
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if not image_info:
        print(f" No valid images found in {folder_path}")
        return

    widths = np.array([w for w, _, _ in image_info])
    heights = np.array([h for _, h, _ in image_info])
    channels = np.array([c for _, _, c in image_info])

    print(f"\n Folder: {folder_path}")
    print(f"총 이미지 수: {len(image_info)}장")
    print(f"이미지 형식 분포: {dict(file_types)}")

    print(f"\n 이미지 크기 통계:")
    print(f"  - 평균 크기: {widths.mean():.1f} x {heights.mean():.1f}")
    print(f"  - 최소 크기: {widths.min()} x {heights.min()}")
    print(f"  - 최대 크기: {widths.max()} x {heights.max()}")

    print(f"\n 채널 수 통계:")
    print(f"  - 평균 채널 수: {channels.mean():.1f}")
    print(f"  - 분포: {Counter(channels)}")        