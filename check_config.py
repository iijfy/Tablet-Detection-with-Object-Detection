# check_config.py
import os, yaml

cfg_p = "retinanet_experiments/configs/retinanet.yaml"
with open(cfg_p, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

checks = {
    "img_root (should exist)": os.path.exists(cfg["paths"]["img_root"]),
    "train_ann (expect False now)": os.path.exists(cfg["paths"]["train_ann"]),
    "val_ann (expect False now)": os.path.exists(cfg["paths"]["val_ann"]),
    "out_dir (will be created)": os.path.exists(cfg["paths"]["out_dir"]),
}

print("\n[CONFIG SUMMARY]")
for k, v in checks.items():
    print(f"{k:30} : {v}")