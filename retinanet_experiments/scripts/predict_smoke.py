# retinanet_experiments/scripts/predict_smoke.py
"""
- retinanet_experiments/configs/retinanet.yaml 경로를 읽고
- 가장 최근 체크포인트를 로드한 뒤
- val.json에서 앞쪽 N장만 추론하여 시각화 PNG로 저장합니다.
"""

import json
from pathlib import Path
import argparse
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
import torchvision
from torchvision.ops import box_convert

def load_cfg():
    cfg_p = Path("retinanet_experiments/configs/retinanet.yaml")
    with open(cfg_p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def find_latest_ckpt(out_dir: Path) -> Path:
    ck = out_dir / "checkpoints"
    cands = sorted(ck.glob("retinanet_epoch*.pth"))
    assert cands, f"체크포인트가 없습니다: {ck}"
    return cands[-1]

def load_categories(val_json: Path):
    with open(val_json, "r", encoding="utf-8") as f:
        j = json.load(f)
    # category id → name 매핑 (없으면 id 문자열로)
    id2name = {int(c["id"]): c.get("name", str(c["id"])) for c in j["categories"]}
    images = j["images"]
    return id2name, images

def draw_boxes(img: Image.Image, boxes_xyxy, labels, id2name):
    draw = ImageDraw.Draw(img)
    # (선택) 폰트 없으면 기본
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = None
    for box, lab in zip(boxes_xyxy, labels):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        name = id2name.get(int(lab), str(int(lab)))
        txt = f"{name}"
        if font:
            draw.text((x1+2, y1+2), txt, fill=(255, 255, 255), font=font)
        else:
            draw.text((x1+2, y1+2), txt, fill=(255, 255, 255))
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=8, help="시각화할 이미지 개수")
    ap.add_argument("--score", type=float, default=0.35, help="confidence threshold")
    args = ap.parse_args()

    cfg = load_cfg()
    paths = cfg["paths"]
    img_root = Path(paths["img_root"])
    val_json = Path(paths["val_ann"])
    out_dir = Path(paths["out_dir"])
    vis_dir = out_dir / "vis_val"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드 (백본만 ImageNet, 헤드는 학습 가중치에서 불러옴)
    ckpt = find_latest_ckpt(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=None,
        weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
        num_classes=147  # 학습과 동일하게 명시
    ).to(device).eval()

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    print(f"[INFO] loaded checkpoint: {ckpt}")

    # 데이터 준비
    id2name, images_meta = load_categories(val_json)
    # 파일 이름을 보장하기 위해 json에서 file_name 사용
    count = 0
    for info in images_meta:
        if count >= args.num:
            break
        p = img_root / info["file_name"]
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        img_t = torchvision.transforms.functional.to_tensor(img).to(device)

        with torch.no_grad():
            pred = model([img_t])[0]

        # 스코어 필터링
        keep = pred["scores"] >= args.score
        boxes = pred["boxes"][keep].detach().cpu().numpy()
        labels = pred["labels"][keep].detach().cpu().numpy()

        # 그리기 & 저장
        vis = img.copy()
        vis = draw_boxes(vis, boxes, labels, id2name)
        save_p = vis_dir / f"val_{info['id']}_{p.stem}.png"
        vis.save(save_p)
        print(f"[VIS] saved: {save_p}")
        count += 1

    print(f"[DONE] visualized {count} images to: {vis_dir}")

if __name__ == "__main__":
    main()