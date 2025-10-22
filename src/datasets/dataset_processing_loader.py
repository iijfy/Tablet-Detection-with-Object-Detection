import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.config import load_config

config = load_config()  # ../../config.yaml을 자동으로 읽음
train_ratio = config["dataset"]["train_ratio"]
batch_size = config["dataset"]["batch_size"]
seed = config["project"]["seed"]


# ==========================================================
# 1️. Seed 고정 함수
# ==========================================================
def set_global_seed(seed=seed):
    """모든 라이브러리의 랜덤 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"===========lobal seed fixed to {seed}===========")


# ==========================================================
# 2. Worker seed 고정 (DataLoader 내부용)
# ==========================================================
def seed_worker(worker_id):
    """DataLoader의 각 worker의 난수 시드를 고정"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==========================================================
# 3. Stratified Split 함수
# ==========================================================
def stratified_split_by_category(df, train_ratio=0.8, seed=42):
    """categories_id 비율을 고려하여 train/test 분할 (object 단위)"""
    np.random.seed(seed)
    train_list, test_list = [], []

    for cat_id, group in df.groupby("categories_id"):
        n_total = len(group)
        n_train = int(n_total * train_ratio)
        indices = np.random.permutation(group.index)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        train_list.append(df.loc[train_idx])
        test_list.append(df.loc[test_idx])

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    print(f"===========분리 완료 (seed={seed}): Train {len(train_df)}개 / Test {len(test_df)}개===========")
    return train_df, test_df


# ==========================================================
# 4. Albumentations Transform 정의
# ==========================================================
def get_train_transform():
    return A.Compose([
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.RandomRotate90(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def get_test_transform():
    return A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


# ==========================================================
# 5. Dataset 클래스 정의
# ==========================================================
class PillDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.grouped = df.groupby("images_file_name")
        self.image_names = list(self.grouped.groups.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        records = self.grouped.get_group(image_name)
        image_path = os.path.join(self.image_dir, image_name)
        image = np.array(Image.open(image_path).convert("RGB"))

        bboxes, labels = [], []
        for _, row in records.iterrows():
            x1, y1 = row["bbox_x"], row["bbox_y"]
            x2, y2 = x1 + row["bbox_w"], y1 + row["bbox_h"]
            bboxes.append([x1, y1, x2, y2])
            labels.append(row["categories_name"])

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["class_labels"]

        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": labels
        }
        return image, target


# ==========================================================
# 6. DataLoader 생성 함수
# ==========================================================
def create_dataloaders(train_df, test_df, image_dir, batch_size= batch_size, num_workers=2, seed=seed):
    """train/test DataLoader를 생성 (seed 고정 포함)"""
    set_global_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset = PillDataset(train_df, image_dir, transform=get_train_transform())
    test_dataset = PillDataset(test_df, image_dir, transform=get_test_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=lambda x: tuple(zip(*x))
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=lambda x: tuple(zip(*x))
    )

    print(f"===========DataLoader 생성 완료 → Train: {len(train_loader)} / Test: {len(test_loader)}===========")
    return train_loader, test_loader

# ==========================================================
# 🔹 Normalize 복원 함수 (denormalization)
# ==========================================================
def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Normalize된 이미지를 다시 원래 색상으로 되돌리는 함수.
    Args:
        tensor (Tensor): [C, H, W]
        mean, std: Normalize 시 사용한 값
    Returns:
        np.ndarray: 복원된 [H, W, C] 이미지 (0~1 범위)
    """
    img = tensor.clone().detach()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    return img


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os

def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Normalize된 이미지를 다시 원래 색상으로 되돌림."""
    img = tensor.clone().detach()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = img.permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)


def visualize_loader_batch(
    data_loader,
    num_images=4,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    normalize_applied=True,
    is_train=False,
    title_prefix=None,
    save_dir=None,
):
    """
    DataLoader의 배치 단위 시각화 (Normalize 해제 + bbox 표시)

    Args:
        data_loader: PyTorch DataLoader
        num_images: 표시할 이미지 수
        mean, std: Normalize 해제 시 사용
        normalize_applied: Normalize 해제 여부
        is_train: True이면 'Train' 제목으로 표시
        title_prefix: 각 이미지의 제목 앞에 붙는 문자열
        save_dir: 지정 시 PNG로 저장
    """
    batch = next(iter(data_loader))
    images, targets = batch

    ncols = 2
    nrows = int(np.ceil(num_images / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 8 * nrows))
    axes = axes.flatten()

    for i in range(num_images):
        if i >= len(images):
            break

        img_tensor = images[i]
        if normalize_applied:
            img = denormalize_image(img_tensor, mean, std)
        else:
            img = img_tensor.permute(1, 2, 0).cpu().numpy()

        boxes = targets[i]["boxes"].cpu().numpy()
        labels = targets[i]["labels"]

        ax = axes[i]
        ax.imshow(img)
        colors = plt.cm.tab10.colors

        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            color = colors[j % len(colors)]
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            label_value = labels[j]
            # Tensor일 경우 숫자 추출
            if isinstance(label_value, torch.Tensor):
                label_value = int(label_value.item())

            ax.text(
                x1,
                y1 - 5,
                str(label_value),
                fontsize=9,
                color="white",
                backgroundcolor=color,
                fontweight="bold",
            )

        title = title_prefix or ("Train" if is_train else "Test")
        ax.set_title(f"{title} Image #{i}", fontsize=10)
        ax.axis("off")

    # 남은 subplot 비활성화
    for k in range(i + 1, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{title_prefix or ('train' if is_train else 'test')}_batch.png"
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"==========={title_prefix or ('Train' if is_train else 'Test')} 시각화 저장 완료 → {path}===========")
    else:
        plt.show()