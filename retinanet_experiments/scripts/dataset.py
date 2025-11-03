import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class SampleRecord:
    image_id: int
    filename: str
    image_path: str
    label_path: Optional[str]


def _load_label_file(
    label_path: str, image_size: Tuple[int, int]
) -> Tuple[List[List[float]], List[int]]:
    """Parse a YOLO-format label file into absolute xyxy boxes and class ids."""
    width, height = image_size
    boxes: List[List[float]] = []
    labels: List[int] = []

    if not label_path or not os.path.exists(label_path):
        return boxes, labels

    with open(label_path, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, xc, yc, w, h = parts

            class_id = int(float(class_id))
            xc = float(xc) * width
            yc = float(yc) * height
            w = float(w) * width
            h = float(h) * height

            if w <= 0 or h <= 0:
                continue

            x1 = max(min(xc - w / 2.0, float(width)), 0.0)
            y1 = max(min(yc - h / 2.0, float(height)), 0.0)
            x2 = max(min(xc + w / 2.0, float(width)), 0.0)
            y2 = max(min(yc + h / 2.0, float(height)), 0.0)

            # Ensure coordinates are ordered after clamping
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            # Drop degenerate boxes
            if (x2 - x1) <= 1e-3 or (y2 - y1) <= 1e-3:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)

    return boxes, labels


def _sanitize_target(target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    boxes = target["boxes"]
    if boxes.numel() == 0:
        return target

    height, width = target["size"].tolist()

    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0.0, float(width))
    boxes[:, 2] = boxes[:, 2].clamp(0.0, float(width))
    boxes[:, 1] = boxes[:, 1].clamp(0.0, float(height))
    boxes[:, 3] = boxes[:, 3].clamp(0.0, float(height))

    x1 = torch.min(boxes[:, 0], boxes[:, 2])
    y1 = torch.min(boxes[:, 1], boxes[:, 3])
    x2 = torch.max(boxes[:, 0], boxes[:, 2])
    y2 = torch.max(boxes[:, 1], boxes[:, 3])

    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    keep = (x2 - x1 > 1e-3) & (y2 - y1 > 1e-3)
    if keep.sum().item() != len(keep):
        boxes = boxes[keep]
        target["labels"] = target["labels"][keep]
        target["iscrowd"] = target["iscrowd"][keep]
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    else:
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    target["boxes"] = boxes
    return target


class TabletDetectionDataset(Dataset):
    """Dataset that reads YOLO-format labels and images for RetinaNet training."""

    def __init__(
        self,
        images_dir: str,
        labels_dir: Optional[str],
        id_map_csv: str,
        split: str,
        transforms: Optional[Callable] = None,
        drop_missing: bool = False,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.split = split

        df = pd.read_csv(id_map_csv)
        df_split = df[df["split"] == split].copy()

        if df_split.empty:
            raise ValueError(f"No entries found for split '{split}' in {id_map_csv}")

        records: List[SampleRecord] = []

        for _, row in df_split.iterrows():
            image_id = int(row["image_id"])
            filename = row["filename"]
            image_path = os.path.join(images_dir, filename)
            label_path = (
                os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
                if labels_dir
                else None
            )

            if not os.path.exists(image_path):
                if drop_missing:
                    continue
                raise FileNotFoundError(
                    f"Image '{image_path}' from split '{split}' not found."
                )

            if label_path and not os.path.exists(label_path) and split != "test":
                # Allow missing labels for test predictions
                if drop_missing:
                    continue

            records.append(
                SampleRecord(
                    image_id=image_id,
                    filename=filename,
                    image_path=image_path,
                    label_path=label_path if label_path and os.path.exists(label_path) else None,
                )
            )

        if not records:
            raise ValueError(
                f"No usable samples discovered for split '{split}'. "
                "Check dataset paths and id_map.csv consistency."
            )

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size

        boxes, labels = _load_label_file(record.label_path, (width, height))

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([record.image_id], dtype=torch.int64),
            "area": (
                (boxes_tensor[:, 2] - boxes_tensor[:, 0])
                * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
                if len(boxes) > 0
                else torch.zeros((0,), dtype=torch.float32)
            ),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        # Ensure boxes remain valid after any transforms.
        target = _sanitize_target(target)

        return image, target


def collate_fn(batch):
    """Dataloader collate_fn that keeps images/targets in lists."""
    return tuple(zip(*batch))
