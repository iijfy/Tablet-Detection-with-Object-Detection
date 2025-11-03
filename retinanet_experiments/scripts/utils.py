import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class AverageMeter:
    name: str
    value: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.value += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.value / self.count if self.count else 0.0


def move_targets_to_device(targets: Iterable[Dict], device: torch.device) -> List[Dict]:
    moved = []
    for target in targets:
        moved_target = {}
        for k, v in target.items():
            if torch.is_tensor(v) and k not in {"image_id", "size"}:
                moved_target[k] = v.to(device)
            else:
                moved_target[k] = v
        moved.append(moved_target)
    return moved


def save_checkpoint(state: Dict, output_dir: str, filename: str) -> str:
    ensure_dir(output_dir)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    return path


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    mins, sec = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"{hrs}h {mins:02d}m {sec:02d}s"
    if mins:
        return f"{mins}m {sec:02d}s"
    return f"{sec}s"


def load_id_map_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].astype(int)
    return df


def load_class_mapping(path: str) -> Tuple[Dict[int, int], Dict[int, str]]:
    df = pd.read_csv(path)
    required_cols = {"yolo_id", "orig_cat_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"class map missing columns: {required_cols - set(df.columns)}")
    yolo_to_kaggle: Dict[int, int] = {}
    yolo_to_name: Dict[int, str] = {}
    for _, row in df.iterrows():
        yolo_id = int(row["yolo_id"])
        class_name = str(row.get("class_name", "")).strip()
        kaggle_id: int
        if class_name and class_name.isdigit():
            kaggle_id = int(class_name)
        else:
            kaggle_id = int(row["orig_cat_id"])
        yolo_to_kaggle[yolo_id] = kaggle_id
        yolo_to_name[yolo_id] = class_name
    return yolo_to_kaggle, yolo_to_name


def clamp_box(box: torch.Tensor, width: int, height: int) -> torch.Tensor:
    x1 = torch.clamp(box[0], 0.0, float(width))
    y1 = torch.clamp(box[1], 0.0, float(height))
    x2 = torch.clamp(box[2], 0.0, float(width))
    y2 = torch.clamp(box[3], 0.0, float(height))
    x1, x2 = torch.min(x1, x2), torch.max(x1, x2)
    y1, y2 = torch.min(y1, y2), torch.max(y1, y2)
    return torch.tensor([x1, y1, x2, y2], dtype=torch.float32)


def filter_by_score(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    max_detections: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = scores >= threshold
    indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
    if max_detections:
        indices = indices[:max_detections]
    return boxes[indices], labels[indices], scores[indices]
