import argparse
import os
import pathlib
import sys
from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(PACKAGE_ROOT))
    from retinanet_experiments.scripts.dataset import TabletDetectionDataset, collate_fn
    from retinanet_experiments.scripts.transforms import build_transforms
    from retinanet_experiments.scripts.utils import (
        clamp_box,
        ensure_dir,
        load_config,
        load_class_mapping,
        load_id_map_csv,
        filter_by_score,
    )
else:
    from .dataset import TabletDetectionDataset, collate_fn
    from .transforms import build_transforms
    from .utils import (
        clamp_box,
        ensure_dir,
        load_config,
        load_class_mapping,
        load_id_map_csv,
        filter_by_score,
    )

from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RetinaNet inference and export Kaggle submission."
    )
    parser.add_argument(
        "--config",
        default="retinanet_experiments/configs/retinanet.yaml",
        help="Path to experiment config.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint file (.pth) with trained weights.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split defined in config data.*_split (e.g. train/val/test).",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Override images directory for inference.",
    )
    parser.add_argument(
        "--labels-dir",
        default=None,
        help="Override labels directory (optional, for evaluation).",
    )
    parser.add_argument(
        "--id-map",
        default=None,
        help="Override id_map CSV path.",
    )
    parser.add_argument(
        "--score-threshold",
        default=None,
        type=float,
        help="Score threshold for filtering predictions.",
    )
    parser.add_argument(
        "--max-detections",
        default=None,
        type=int,
        help="Maximum detections per image.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (cpu or cuda).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV file path.",
    )
    return parser.parse_args()


def build_model(
    num_classes: int,
    pretrained: bool = False,
    pretrained_backbone: bool = True,
) -> torch.nn.Module:
    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None

    if weights is not None and num_classes != len(weights.meta["categories"]):
        weights = None
        if weights_backbone is None:
            weights_backbone = ResNet50_Weights.DEFAULT

    model = retinanet_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
    )
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    max_detections: int | None,
    yolo_to_kaggle: Dict[int, int],
) -> List[Dict]:
    model.eval()
    predictions: List[Dict] = []
    annotation_id = 1

    for images, targets in tqdm(data_loader, desc="Predict", leave=False):
        sizes = [t["size"] for t in targets]
        image_ids = [int(t["image_id"].item()) for t in targets]

        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, size, image_id in zip(outputs, sizes, image_ids):
            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()

            boxes, labels, scores = filter_by_score(
                boxes, labels, scores, score_threshold, max_detections
            )

            height = int(size[0].item())
            width = int(size[1].item())

            for box, label, score in zip(boxes, labels, scores):
                clamped = clamp_box(box, width, height)
                x1, y1, x2, y2 = clamped.tolist()

                w = float(x2 - x1)
                h = float(y2 - y1)
                if w <= 0 or h <= 0:
                    continue

                kaggle_cat = yolo_to_kaggle.get(int(label.item()))
                if kaggle_cat is None:
                    raise KeyError(
                        f"Missing category mapping for predicted label {int(label.item())}"
                    )

                bbox_x = round(float(x1), 2)
                bbox_y = round(float(y1), 2)
                bbox_w = round(w, 2)
                bbox_h = round(h, 2)
                score_val = round(float(score.item()), 6)

                predictions.append(
                    {
                        "annotation_id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(kaggle_cat),
                        "bbox_x": bbox_x,
                        "bbox_y": bbox_y,
                        "bbox_w": bbox_w,
                        "bbox_h": bbox_h,
                        "score": score_val,
                    }
                )
                annotation_id += 1

    return predictions


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    infer_cfg = cfg.get("inference", {})

    split = (
        args.split
        if args.split
        else data_cfg.get("predict_split", data_cfg.get("val_split", "val"))
    )

    split_key = split
    images_dir = args.images_dir or paths_cfg.get(f"{split_key}_images")
    if not images_dir:
        raise ValueError(
            f"Images directory for split '{split}' missing. "
            "Set paths.<split>_images in config or provide --images-dir."
        )
    labels_dir = args.labels_dir or paths_cfg.get(f"{split_key}_labels")

    if args.id_map:
        id_map_path = args.id_map
    else:
        split_specific_map = paths_cfg.get(f"{split_key}_id_map")
        if split_specific_map and os.path.exists(split_specific_map):
            id_map_path = split_specific_map
        else:
            id_map_path = paths_cfg.get("id_map")
    if not id_map_path:
        raise ValueError("id_map path must be provided through config or --id-map.")
    class_map_path = paths_cfg.get("class_map", "yolo/metadata/class_map.csv")
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(
            f"Class map CSV not found at {class_map_path}. "
            "Set paths.class_map in the config or provide --class-map."
        )

    id_map_df = load_id_map_csv(id_map_path)

    device_name = (
        args.device
        if args.device
        else infer_cfg.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    device = torch.device(device_name)

    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else infer_cfg.get("score_threshold", 0.05)
    )
    max_detections = (
        args.max_detections
        if args.max_detections is not None
        else infer_cfg.get("max_detections", 300)
    )

    base_out_dir = paths_cfg.get(
        "out_dir", "retinanet_experiments/outputs/default_run"
    )
    submission_dir = os.path.join(base_out_dir, "submission")
    default_output = os.path.join(
        submission_dir, "submission_retinanet_kaggle.csv"
    )
    output_path = args.output if args.output else default_output
    ensure_dir(os.path.dirname(output_path))

    print(f"[Predict] Split            : {split}")
    print(f"[Predict] Images dir       : {images_dir}")
    print(f"[Predict] Labels dir       : {labels_dir or 'N/A'}")
    print(f"[Predict] id_map           : {id_map_path}")
    print(f"[Predict] Device           : {device}")
    print(f"[Predict] Score threshold  : {score_threshold}")
    print(f"[Predict] Max detections   : {max_detections}")
    print(f"[Predict] Checkpoint       : {args.checkpoint}")
    print(f"[Predict] Output CSV       : {output_path}")
    if "split" in id_map_df.columns:
        num_entries = int((id_map_df["split"] == split).sum())
        print(f"[Predict] id_map entries  : {num_entries}")

    dataset = TabletDetectionDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        id_map_csv=id_map_path,
        split=split,
        transforms=build_transforms(train=False),
        drop_missing=True,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=infer_cfg.get("batch_size", 2),
        shuffle=False,
        num_workers=infer_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
    )

    num_classes = model_cfg.get("num_classes")
    if num_classes is None:
        raise ValueError("model.num_classes must be defined in config.")

    yolo_to_kaggle, _ = load_class_mapping(class_map_path)

    model = build_model(
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", False),
        pretrained_backbone=model_cfg.get("pretrained_backbone", True),
    )
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    model.to(device)

    predictions = run_inference(
        model=model,
        data_loader=data_loader,
        device=device,
        score_threshold=score_threshold,
        max_detections=max_detections,
        yolo_to_kaggle=yolo_to_kaggle,
    )

    valid_image_ids = set(id_map_df["image_id"].astype(int).tolist())
    invalid_ids = {
        pred["image_id"]
        for pred in predictions
        if pred["image_id"] not in valid_image_ids
    }
    if invalid_ids:
        raise ValueError(
            f"Predictions contain image_ids not listed in id_map: {sorted(invalid_ids)[:5]}"
        )

    columns = [
        "annotation_id",
        "image_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "score",
    ]
    df = pd.DataFrame(predictions, columns=columns)

    print(f"[Verify] Columns          : {df.columns.tolist()}")
    print(f"[Verify] Num detections   : {len(df)}")
    if len(df) > 0:
        print("[Verify] Sample rows:")
        print(df.head(3).to_string(index=False))
    else:
        print("[Verify] Sample rows      : (no detections)")

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} predictions to {output_path}")


if __name__ == "__main__":
    main()
