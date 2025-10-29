# -*- coding: utf-8 -*-
"""
데이터 격차(Data Gap) 리포트 😊
- 입력: yolo/reports/class_distribution_overall.csv, metadata/class_map.csv
- 출력: yolo/reports/data_gap_plan.csv (부족 클래스와 필요한 추가 인스턴스/이미지 수 추정)
- 아이디어: 타깃 바닥선 = 전체 분포의 'median'(또는 P75의 0.6배) 중 큰 값
"""

import os, math, csv
import pandas as pd

# 📁 경로 잡기(네 폴더 구조 맞춤)
ROOT = os.path.dirname(os.path.dirname(__file__))         # .../yolo
REPORTS = os.path.join(ROOT, "reports")
META = os.path.join(ROOT, "metadata")

DIST_CSV = os.path.join(REPORTS, "class_distribution_overall.csv")
CMAP_CSV = os.path.join(META, "class_map.csv")
OUT_CSV  = os.path.join(REPORTS, "data_gap_plan.csv")

def info(*a): print("[INFO]", *a)
os.makedirs(REPORTS, exist_ok=True)

# 1) 데이터 로드
dist = pd.read_csv(DIST_CSV)  # columns: category_id,count
cmap = pd.read_csv(CMAP_CSV)  # columns 예: orig_cat_id,yolo_id,class_name (중복 있을 수 있음)

# 2) 클래스맵 정리(혹시 중복 orig_cat_id가 있다면 첫 행만 사용)
if "orig_cat_id" in cmap.columns:
    cmap = cmap.drop_duplicates(subset=["orig_cat_id"], keep="first")
    cmap = cmap[["orig_cat_id","class_name"]].rename(columns={"orig_cat_id":"category_id"})
elif "category_id" in cmap.columns:
    cmap = cmap.drop_duplicates(subset=["category_id"], keep="first")
    cmap = cmap[["category_id","class_name"]]
else:
    # 헤더가 예외적일 경우 대비
    cmap.columns = [c.strip() for c in cmap.columns]
    cmap = cmap.rename(columns={cmap.columns[0]:"category_id", cmap.columns[1]:"class_name"})
    cmap = cmap.drop_duplicates(subset=["category_id"], keep="first")

# 3) 조인하여 사람이 읽기 쉬운 표로
df = dist.merge(cmap, on="category_id", how="left")
df["class_name"] = df["class_name"].fillna("(unknown)")

# 4) 통계로 타깃 바닥선 결정
median = float(df["count"].median())
p75 = float(df["count"].quantile(0.75))
target_floor = max(int(median), int(p75 * 0.6), 50)  # 너무 작지 않게 하한 50 설정

# 5) 부족량 계산
df["need_instances"] = (target_floor - df["count"]).clip(lower=0)
# 이미지 수 추정: 알약은 보통 1~2개/이미지라고 가정 → 보수적으로 1.2개/이미지
per_image = 1.2
df["est_new_images"] = df["need_instances"].apply(lambda x: int(math.ceil(x / per_image)) if x>0 else 0)

# 6) 정렬 및 저장
df = df.sort_values(["need_instances","count"], ascending=[False, True])
cols = ["category_id","class_name","count","need_instances","est_new_images"]
df_out = df[cols]
df_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
info(f"✅ 저장 완료: {OUT_CSV}")

# 7) 콘솔 요약(Top-15)
print("\n[TOP-15 부족 클래스]")
print(df_out.head(15).to_string(index=False))

# 8) 전체 통계 인쇄
print("\n[분포 요약]")
print(f" - classes: {len(df)}")
print(f" - total instances: {int(df['count'].sum())}")
print(f" - min/median/mean/max: {int(df['count'].min())}/{int(median)}/{df['count'].mean():.1f}/{int(df['count'].max())}")
print(f" - target_floor: {target_floor}")
print(f" - 부족 클래스 수: {(df['need_instances']>0).sum()}")
print(f" - 총 필요 인스턴스: {int(df['need_instances'].sum())}")
print(f" - 예상 추가 이미지 수(≈×{per_image}): {int(df['est_new_images'].sum())}")