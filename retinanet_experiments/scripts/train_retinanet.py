# retinanet_experiments/scripts/train_retinanet.py
"""
포인트
- num_classes = (배경 제외) 자동 산출.
- COCO json의 bbox가 0~1 스케일(xywh)인지, 픽셀(xywh)인지 자동 감지 후 픽셀 xyxy로 변환.
- 1 epoch 스모크 학습으로 동작 확인, 체크포인트 저장.
- torchvision 0.17 / torch 2.2 API 규칙에 맞는 weights/weights_backbone 사용.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms.functional import to_tensor
from torchvision.ops import box_convert
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from PIL import Image
import yaml


# -------------------------------
# 0) 유틸
# -------------------------------
def _maybe_denorm_xywh(x, y, w, h, W, H):
    """bbox가 0~1 범위면 픽셀 좌표로 변환, 이미 픽셀이면 그대로 반환."""
    if max(x, y, w, h) <= 1.000001:  # 여유치 포함
        return x * W, y * H, w * W, h * H
    return x, y, w, h


def _make_cat_mapping(categories):
    """카테고리 id가 1..N 연속이 아닐 수 있으므로 연속 idx(1..N)로 매핑."""
    cat_ids = [c["id"] for c in categories]
    cat_ids_sorted = sorted(set(cat_ids))
    cat_to_idx = {cid: i + 1 for i, cid in enumerate(cat_ids_sorted)}  # 1부터 시작(배경 제외)
    return cat_to_idx, len(cat_to_idx)


def collate_fn(batch):
    """DataLoader용 collate: None 샘플 제거."""
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))


# -------------------------------
# 1) Dataset
# -------------------------------
class CocoDet(torch.utils.data.Dataset):
    """
    COCO Detection 형식 Dataset.
    - images: [{id, file_name, width, height}]
    - annotations: [{image_id, bbox=[x,y,w,h], category_id, ...}]
    - categories: [{id, name}, ...] (id가 반드시 연속이진 않다)
    """

    def __init__(self, img_root, ann_json):
        self.img_root = Path(img_root)

        with open(ann_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # 이미지 메타
        self.images = {img["id"]: img for img in coco["images"]}
        self.ids = [img["id"] for img in coco["images"]]

        # 카테고리 매핑 (1..num_classes)
        self.cat_to_idx, self.num_classes = _make_cat_mapping(coco["categories"])

        # 이미지별 어노테이션 묶기 (crowd 제외)
        per_img = {}
        for a in coco["annotations"]:
            if a.get("iscrowd", 0) == 1:
                continue
            per_img.setdefault(a["image_id"], []).append(a)
        self.per_img = per_img

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        info = self.images[img_id]
        p = self.img_root / info["file_name"]
        if not p.exists():
            # 누락된 샘플은 학습 중단보다 skip이 안전
            return None

        # 이미지 로드
        img = Image.open(p).convert("RGB")
        W, H = img.size
        img_t = to_tensor(img)

        # 어노테이션 -> 픽셀 xyxy 변환
        anns = self.per_img.get(img_id, [])
        boxes_xywh = []
        labels = []
        for a in anns:
            x, y, w, h = a["bbox"]  # COCO 규격상 xywh
            x, y, w, h = _maybe_denorm_xywh(x, y, w, h, W, H)
            if w <= 0 or h <= 0:
                continue
            boxes_xywh.append([x, y, w, h])
            # 카테고리 id를 1..N 연속 idx로 변환
            labels.append(self.cat_to_idx[int(a["category_id"])])

        if len(boxes_xywh) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes_xywh, dtype=torch.float32)
            boxes = box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([img_id])}
        return img_t, target


# -------------------------------
# 2) Train loop
# -------------------------------
def train_one_epoch(model, loader, optimizer, device, epoch, print_freq=50):
    model.train()
    running, n = 0.0, 0
    t0 = time.time()

    for it, (imgs, targets) in enumerate(loader, 1):
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # torchvision detection API: 학습 시 손실 dict 반환
        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += float(loss)
        n += 1
        if it % print_freq == 0:
            print(f"[epoch {epoch} {it:4d}/{len(loader)}] loss={running/n:.4f}  ({time.time()-t0:.1f}s)")
            running, n, t0 = 0.0, 0, time.time()


# -------------------------------
# 3) Main
# -------------------------------
def main():
    # (A) 인자
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    # (B) 설정 로드
    cfg_p = Path("retinanet_experiments/configs/retinanet.yaml")
    cfg = yaml.safe_load(open(cfg_p, "r", encoding="utf-8"))

    paths = cfg["paths"]
    img_root = paths["img_root"]
    train_json = paths["train_ann"]
    val_json = paths["val_ann"]
    out_dir = Path(paths["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # (C) 데이터
    train_ds = CocoDet(img_root, train_json)
    val_ds = CocoDet(img_root, val_json)
    num_classes = train_ds.num_classes  # RetinaNet은 배경 제외 개수

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (D) 모델
    #  - 헤드(분류 레이어)는 num_classes로 새로 초기화 → weights=None
    #  - 백본은 ImageNet으로 프리트레인된 ResNet50 사용 → weights_backbone=ResNet50_Weights.DEFAULT
    model = retinanet_resnet50_fpn_v2(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
    ).to(device)

    # (E) 옵티마이저/스케줄러
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"])
    )
    milestones = [int(m) for m in cfg["train"]["lr_steps"]]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # (F) 요약 출력
    print(f"[CFG] num_classes={num_classes}, epochs={args.epochs}, batch={args.batch}, workers={args.workers}")
    print(f"[PATH] img_root={img_root}")
    print(f"[PATH] train_json={train_json}")
    print(f"[PATH] val_json={val_json}")
    print(f"[PATH] out_dir={out_dir}")

    # (G) 학습
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch, print_freq=50)
        scheduler.step()

        # quick forward check on val (작동 확인용)
        model.eval()
        with torch.no_grad():
            for imgs, _ in val_loader:
                _ = model([im.to(device) for im in imgs])
                break

        # (H) 체크포인트 저장
        ckpt = ckpt_dir / f"retinanet_epoch{epoch:02d}.pth"
        torch.save(
            {"epoch": epoch, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
            ckpt
        )
        print(f"[epoch {epoch}] saved -> {ckpt}")


if __name__ == "__main__":
    main()