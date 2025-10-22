# src/config.py
import os
import yaml

def load_config(config_path=None):
    """항상 project_root/config.yaml을 읽도록 절대경로 기반 설정"""
    if config_path is None:
        # 현재 파일 (src/config.py)의 절대 경로 기준
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "../config.yaml")  # src 기준 한 단계 위
        config_path = os.path.abspath(config_path)                 # 절대경로 변환

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config