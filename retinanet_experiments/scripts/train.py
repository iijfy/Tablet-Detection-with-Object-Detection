import argparse
import os
import pathlib
import sys
import time
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)
from tqdm import tqdm

if __package__ is None or __package__ == "":
    PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(PACKAGE_ROOT))
    from retinanet_experiments.scripts.dataset import TabletDetectionDataset, collate_fn
    from retinanet_experiments.scripts.transforms import build_transforms
    from retinanet_experiments.scripts.utils import (
        AverageMeter,
        ensure_dir,
        format_seconds,
        load_config,
        move_targets_to_device,
        save_checkpoint,
        set_seed,
    )
else:
    from .dataset import TabletDetectionDataset, collate_fn
    from .transforms import build_transforms
    from .utils import (
        AverageMeter,
        ensure_dir,
        format_seconds,
        load_config,
        move_targets_to_device,
        save_checkpoint,
        set_seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RetinaNet on tablet dataset.")
    parser.add_argument(
        "--config",
        default="retinanet_experiments/configs/retinanet.yaml",
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device from config (cpu or cuda).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config.",
    )
    return parser.parse_args()


def build_model(model_cfg: Dict) -> torch.nn.Module:
    num_classes = model_cfg.get("num_classes")
    if num_classes is None:
        raise ValueError("model.num_classes must be specified in the config.")

    pretrained = model_cfg.get("pretrained", True)
    pretrained_backbone = model_cfg.get("pretrained_backbone", True)
    trainable_backbone_layers = model_cfg.get("trainable_backbone_layers", 3)

    weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None

    if weights is not None and num_classes != len(weights.meta["categories"]):
        # torchvision enforces matching class counts when full weights are used.
        # Fallback to ImageNet-pretrained backbone and random detection head.
        print(
            "[Model] num_classes mismatch with pretrained weights; "
            "falling back to pretrained backbone only."
        )
        weights = None
        if weights_backbone is None:
            weights_backbone = ResNet50_Weights.DEFAULT

    model = retinanet_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=weights_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
        num_classes=num_classes,
    )

    return model


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    clip_grad_norm: float = 0.0,
) -> float:
    model.train()
    loss_meter = AverageMeter("loss")

    for images, targets in tqdm(data_loader, desc="Train", leave=False):
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()

        if clip_grad_norm and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad_norm
            )

        optimizer.step()

        loss_meter.update(losses.item(), n=len(images))

    return loss_meter.avg


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    prev_state = model.training
    prev_grad_state = torch.is_grad_enabled()
    model.train()
    torch.set_grad_enabled(False)
    loss_meter = AverageMeter("val_loss")

    for images, targets in tqdm(data_loader, desc="Val  ", leave=False):
        images = [img.to(device) for img in images]
        targets = move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_meter.update(losses.item(), n=len(images))

    torch.set_grad_enabled(prev_grad_state)
    if not prev_state:
        model.eval()
    else:
        model.train()

    return loss_meter.avg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    paths_cfg = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    device_name = (
        args.device
        if args.device
        else train_cfg.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    device = torch.device(device_name)

    output_dir = args.output_dir or paths_cfg.get("out_dir", "retinanet_experiments/outputs/default_run")
    ensure_dir(output_dir)

    print(f"[Config] Using device: {device}")
    print(f"[Config] Output dir : {output_dir}")
    print(f"[Config] Seed       : {seed}")

    batch_size = train_cfg.get("batch_size", 4)
    num_workers = train_cfg.get("num_workers", 4)
    epochs = train_cfg.get("epochs", 20)
    lr = train_cfg.get("lr", 0.0005)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    clip_grad_norm = train_cfg.get("clip_grad_norm", 0.0)
    lr_steps: List[int] = train_cfg.get("lr_steps", [15, 20])
    lr_gamma = train_cfg.get("lr_gamma", 0.1)

    train_dataset = TabletDetectionDataset(
        images_dir=paths_cfg["train_images"],
        labels_dir=paths_cfg.get("train_labels"),
        id_map_csv=paths_cfg["id_map"],
        split=data_cfg.get("train_split", "train"),
        transforms=build_transforms(train=True, enable_hflip=data_cfg.get("hflip", True)),
    )

    val_dataset = TabletDetectionDataset(
        images_dir=paths_cfg["val_images"],
        labels_dir=paths_cfg.get("val_labels"),
        id_map_csv=paths_cfg["id_map"],
        split=data_cfg.get("val_split", "val"),
        transforms=build_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    model = build_model(model_cfg)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov=train_cfg.get("nesterov", False),
        )
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=lr_steps,
        gamma=lr_gamma,
    )

    best_val_loss = float("inf")
    history = []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")

        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, clip_grad_norm
        )
        val_loss = evaluate(model, val_loader, device)
        lr_scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        }
        history.append(metrics)

        elapsed = format_seconds(time.time() - start_time)
        print(
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={current_lr:.6f} | elapsed={elapsed}"
        )

        checkpoint_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "lr_scheduler_state": lr_scheduler.state_dict(),
            "metrics": metrics,
            "config_path": args.config,
        }

        save_checkpoint(
            checkpoint_state,
            output_dir,
            f"checkpoint_epoch_{epoch:03d}.pth",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = save_checkpoint(
                checkpoint_state,
                output_dir,
                "best_model.pth",
            )
            print(f"[Best] Updated best checkpoint: {best_path}")

    total_elapsed = format_seconds(time.time() - start_time)
    print(f"\nTraining complete in {total_elapsed}")

    history_path = os.path.join(output_dir, "history.pt")
    torch.save(history, history_path)
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
