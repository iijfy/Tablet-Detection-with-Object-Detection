"""Ensemble RetinaNet and YOLO predictions into a Kaggle-ready submission CSV."""

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse RetinaNet & YOLO predictions (Weighted Box Fusion)."
    )
    parser.add_argument(
        "--retinanet",
        default="retinanet_experiments/outputs/retinanet_resnet50_fpn_v2/submission/submission_retinanet_kaggle.csv",
        help="Path to RetinaNet submission CSV.",
    )
    parser.add_argument(
        "--yolo",
        default="yolo/submission_yolo_v8l_ver1.csv",
        help="Path to YOLO submission CSV.",
    )
    parser.add_argument(
        "--output",
        default="retinanet_experiments/outputs/retinanet_resnet50_fpn_v2/submission/submission_ensemble_wbf.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--weights",
        nargs=2,
        type=float,
        default=(1.0, 1.0),
        metavar=("RETINA_W", "YOLO_W"),
        help="Model weights used during fusion (RetinaNet, YOLO).",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.55,
        help="IoU threshold for merging boxes.",
    )
    parser.add_argument(
        "--skip-box-thr",
        type=float,
        default=0.001,
        help="Discard detections below this raw score before fusion.",
    )
    parser.add_argument(
        "--min-fused-score",
        type=float,
        default=0.05,
        help="Drop fused boxes below this score.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=None,
        help="Optional max detections kept per image after fusion.",
    )
    return parser.parse_args()


def xywh_to_xyxy(row: pd.Series) -> Tuple[float, float, float, float]:
    x1 = float(row["bbox_x"])
    y1 = float(row["bbox_y"])
    w = float(row["bbox_w"])
    h = float(row["bbox_h"])
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2


def xyxy_to_xywh(box: np.ndarray) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.tolist()
    x1_i = int(round(max(0.0, x1)))
    y1_i = int(round(max(0.0, y1)))
    x2_i = int(round(max(x1_i + 1, x2)))
    y2_i = int(round(max(y1_i + 1, y2)))
    w = max(1, int(round(x2_i - x1_i)))
    h = max(1, int(round(y2_i - y1_i)))
    return x1_i, y1_i, w, h


def calc_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, (xa2 - xa1)) * max(0.0, (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1)) * max(0.0, (yb2 - yb1))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def weighted_box_fusion(
    all_boxes: Sequence[Sequence[np.ndarray]],
    all_scores: Sequence[Sequence[float]],
    all_labels: Sequence[Sequence[int]],
    weights: Sequence[float],
    iou_thr: float,
    skip_box_thr: float,
) -> List[Tuple[np.ndarray, int, float]]:
    candidates: List[Dict] = []
    for boxes, scores, labels, weight in zip(all_boxes, all_scores, all_labels, weights):
        if weight <= 0:
            continue
        for box, score, label in zip(boxes, scores, labels):
            if score < skip_box_thr:
                continue
            candidates.append(
                {
                    "box": np.asarray(box, dtype=np.float32),
                    "score": float(score),
                    "label": int(label),
                    "weight": float(weight),
                }
            )

    candidates.sort(key=lambda x: x["score"] * x["weight"], reverse=True)

    clusters: List[Dict] = []
    for cand in candidates:
        matched = False
        for cluster in clusters:
            if cluster["label"] != cand["label"]:
                continue
            if calc_iou(cluster["box"], cand["box"]) >= iou_thr:
                w = cand["score"] * cand["weight"]
                cluster["box_sum"] += cand["box"] * w
                cluster["weight_sum"] += w
                cluster["score_sum"] += cand["score"] * cand["weight"]
                cluster["weight_only_sum"] += cand["weight"]
                cluster["max_score"] = max(cluster["max_score"], cand["score"])
                matched = True
                break
        if not matched:
            w = cand["score"] * cand["weight"]
            clusters.append(
                {
                    "label": cand["label"],
                    "box": cand["box"].copy(),
                    "box_sum": cand["box"] * w,
                    "weight_sum": w,
                    "score_sum": cand["score"] * cand["weight"],
                    "weight_only_sum": cand["weight"],
                    "max_score": cand["score"],
                }
            )

    fused: List[Tuple[np.ndarray, int, float]] = []
    for cluster in clusters:
        if cluster["weight_sum"] <= 0:
            continue
        fused_box = cluster["box_sum"] / cluster["weight_sum"]
        score_weight_avg = (
            cluster["score_sum"] / cluster["weight_only_sum"]
            if cluster["weight_only_sum"] > 0
            else cluster["max_score"]
        )
        fused_score = max(cluster["max_score"], score_weight_avg)
        fused.append((fused_box, cluster["label"], float(fused_score)))
    return fused


def fuse_predictions(
    retina_df: pd.DataFrame,
    yolo_df: pd.DataFrame,
    weights: Tuple[float, float],
    iou_thr: float,
    skip_box_thr: float,
    min_fused_score: float,
    max_detections: int | None,
) -> pd.DataFrame:
    dfs = [retina_df, yolo_df]
    fused_rows = []

    retina_groups = retina_df.groupby("image_id", sort=False)
    yolo_groups = yolo_df.groupby("image_id", sort=False)
    image_ids = sorted(set(retina_groups.groups.keys()) | set(yolo_groups.groups.keys()))

    for image_id in image_ids:
        per_model_boxes: List[List[np.ndarray]] = []
        per_model_scores: List[List[float]] = []
        per_model_labels: List[List[int]] = []

        for df, groups in zip(dfs, [retina_groups, yolo_groups]):
            if image_id not in groups.groups:
                per_model_boxes.append([])
                per_model_scores.append([])
                per_model_labels.append([])
                continue
            rows = groups.get_group(image_id)
            boxes = [np.asarray(xywh_to_xyxy(row), dtype=np.float32) for _, row in rows.iterrows()]
            scores = [float(row["score"]) for _, row in rows.iterrows()]
            labels = [int(row["category_id"]) for _, row in rows.iterrows()]
            per_model_boxes.append(boxes)
            per_model_scores.append(scores)
            per_model_labels.append(labels)

        fused = weighted_box_fusion(
            per_model_boxes,
            per_model_scores,
            per_model_labels,
            weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        fused = [
            (box, label, score)
            for box, label, score in fused
            if score >= min_fused_score
        ]

        if max_detections:
            fused.sort(key=lambda x: x[2], reverse=True)
            fused = fused[:max_detections]

        for fused_box, label, score in fused:
            bbox_x, bbox_y, bbox_w, bbox_h = xyxy_to_xywh(fused_box)
            fused_rows.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox_x": bbox_x,
                    "bbox_y": bbox_y,
                    "bbox_w": bbox_w,
                    "bbox_h": bbox_h,
                    "score": round(float(score), 6),
                }
            )

    result_df = pd.DataFrame(fused_rows)
    result_df.sort_values(["image_id", "score"], ascending=[True, False], inplace=True)
    return result_df


def main() -> None:
    args = parse_args()

    retina_path = Path(args.retinanet)
    yolo_path = Path(args.yolo)
    if not retina_path.exists():
        raise FileNotFoundError(f"RetinaNet submission not found: {retina_path}")
    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO submission not found: {yolo_path}")

    retina_df = pd.read_csv(retina_path)
    yolo_df = pd.read_csv(yolo_path)

    required_cols = [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
    ]
    for df, name in [(retina_df, "RetinaNet"), (yolo_df, "YOLO")]:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")

    retina_df = retina_df.copy()
    yolo_df = yolo_df.copy()
    retina_df.drop(columns=["annotation_id"], inplace=True, errors="ignore")
    yolo_df.drop(columns=["annotation_id"], inplace=True, errors="ignore")

    weights = tuple(float(w) for w in args.weights)
    fused_df = fuse_predictions(
        retina_df,
        yolo_df,
        weights=weights,
        iou_thr=args.iou_thr,
        skip_box_thr=args.skip_box_thr,
        min_fused_score=args.min_fused_score,
        max_detections=args.max_detections,
    )

    fused_df.insert(0, "annotation_id", range(1, len(fused_df) + 1))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fused_df.to_csv(output_path, index=False)

    print("[Ensemble] Columns   :", fused_df.columns.tolist())
    print("[Ensemble] Rows      :", len(fused_df))
    print("[Ensemble] YOLO in   :", len(yolo_df))
    print("[Ensemble] Retina in :", len(retina_df))
    if len(fused_df) > 0:
        print("[Ensemble] Sample rows:\n", fused_df.head(5).to_string(index=False))
    print(f"[Ensemble] Saved to  : {output_path}")


if __name__ == "__main__":
    main()
