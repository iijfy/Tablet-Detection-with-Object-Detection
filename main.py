# src/main.py
import pandas as pd



import argparse
from src.datasets.dataset_initial_view import (
    copy_all_json_files,
    flatten_json_section,
    load_json_to_dataframe,
    load_image_filenames_to_df,
    compare_filenames,
    expand_bbox_columns,
    validate_bbox,
    visualize_bbox_grid_from_image_df,
    analyze_image_folder,
)
from src.datasets.dataset_processing_loader import *

import torch
from torch.utils.data import DataLoader
from src.datasets.dataset_processing_loader import (
    PillDataset,
    get_train_transform,
    get_test_transform,
    stratified_split_by_category
)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'     # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

def main():

#------------------------------------ --------- 
#  1. 데이어 확인
#------------------------------------ --------- 

#---------1.1 데이터 초기 처리 ---------
    # json파일 하나의 폴더에 저장(1회 수행))
    #copy_all_json_files("./datafile/train_annotations", "./datafile/total_json")

#---------1.2 메타 데이터 데이터프레임 생성 및 csv 저장 ---------
    # json 파일을 DataFrame으로 로드
    #folder = "./datafile/total_json"
    #train_anno_df = load_json_to_dataframe(folder, df_name="train_anno_df")
    #print(train_anno_df.head())
    #train_anno_df.to_csv("./datafile/train_anno_data.csv", index=False)
    ## image 파일명을 DataFrame으로 로드
    #folder = "./datafile/train_images"
    #train_images_df = load_image_filenames_to_df(folder, df_name="train_images_df")
    #print(train_images_df.head())
    #train_images_df.to_csv("./datafile/train_images_list.csv", index=False)

#---------1.3 저장된 메타데이터 불러오기 및 검증 결과 저장 ---------
    #train_anno_df = pd.read_csv("./datafile/train_anno_data.csv")
    #train_images_df = pd.read_csv("./datafile/train_images_list.csv")
    #result = compare_filenames(train_anno_df, train_images_df)
    #train_anno_df = expand_bbox_columns(
    #train_anno_df[["images_file_name", "annotations_id", "annotations_bbox", "categories_id", "categories_name"]])
    #result = validate_bbox(train_anno_df)
    #train_anno_df = train_anno_df.sort_values(by="images_file_name").reset_index(drop=True)
    #train_anno_df.to_csv("./dataset/train_metadata.csv", index=False)
    #print("데이터 처리 완료.")

#---------1.4 시각화
    train_anno_df = pd.read_csv("./datafile/train_metadata.csv")
    train_images_df = pd.read_csv("./datafile/train_images_list.csv")
    print("===========원본 이미지 시각화===========")
    visualize_bbox_grid_from_image_df(train_images_df, train_anno_df, start=0, end=8, show=True) 

 #---------1.5 이미지 속성확인
    print("===========이미지 속성 확인===========")
    base_dir = "datafile"
    train_dir = os.path.join(base_dir, "train_images")
    test_dir = os.path.join(base_dir, "test_images")
    print("===  이미지 데이터셋 분석 결과 ===")
    analyze_image_folder(train_dir)
    analyze_image_folder(test_dir)      



#------------------------------------ --------- 
#  2. 전처 및 Dataloader 구성 파이프라인
#------------------------------------ --------- 

    train_anno_df = pd.read_csv("annotations.csv")
    print(train_anno_df)
    
    image_dir = "./datafile/train_images"
 
    # class 비율 고려하여 분리
    train_df, test_df = stratified_split_by_category(train_anno_df, train_ratio=0.8)
 
    # Dataset 생성
    train_dataset = PillDataset(train_df, image_dir, transform=get_train_transform())
    test_dataset = PillDataset(test_df, image_dir, transform=get_test_transform())
 
    print("===========Dataloader 구성===========")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
 
    print(f"Train Loader: {len(train_loader)} batch, Test Loader: {len(test_loader)} batch")

    print("===========train_loader 시각화===========")
    visualize_loader_batch(train_loader, num_images=4, normalize_applied=True, title_prefix="Train")

    print("===========test_loader 시각화===========")
    visualize_loader_batch(test_loader, num_images=4, normalize_applied=True, title_prefix="Test")


#--------- main ------
if __name__ == "__main__":
    main()