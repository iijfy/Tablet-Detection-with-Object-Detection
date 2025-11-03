# RetinaNet Experiments

This directory contains a reproducible pipeline for training a RetinaNet detector
on the tablet dataset and exporting Kaggle-ready submission files.

## Structure

- `configs/retinanet.yaml` – central configuration for data paths, model and training hyper-parameters.
- `scripts/train.py` – training entrypoint (supports resume & config overrides).
- `scripts/infer.py` – inference/export script that loads trained checkpoints and writes predictions to CSV.
- `scripts/dataset.py` – PyTorch dataset wrapping the YOLO-format labels.
- `scripts/transforms.py` – lightweight image/target transforms.
- `scripts/utils.py` – shared helpers (config loading, checkpoint IO, etc.).

## Quickstart

1. Review and, if necessary, update `configs/retinanet.yaml` so the paths match your workspace.
2. Train the model:

   ```bash
   python -m retinanet_experiments.scripts.train --config retinanet_experiments/configs/retinanet.yaml
   ```

3. Export predictions (defaults to the validation split for sanity checks):

   ```bash
   python -m retinanet_experiments.scripts.infer \
       --config retinanet_experiments/configs/retinanet.yaml \
       --checkpoint retinanet_experiments/outputs/retinanet_resnet50_fpn_v2/best_model.pth \
       --output retinanet_experiments/outputs/retinanet_resnet50_fpn_v2/predictions_val.csv
   ```

   Adjust `--split`, `--images-dir`, and other flags if you need to target a different set (e.g. Kaggle test images).

## Notes

- Training uses `torchvision`'s `retinanet_resnet50_fpn_v2` with COCO-pretrained backbone by default.
- The dataset loader reads YOLO-format label files and computes absolute bounding boxes on the fly.
- Validation loss is tracked each epoch and the best-performing checkpoint is stored as `best_model.pth`.
- The inference script writes predictions in the format expected by the competition submission (`annotation_id` is generated sequentially).
