# scripts/00_check_env.py
import platform
import torch

def main():
    print("===== Environment =====")
    print("Python:", platform.python_version())
    try:
        import ultralytics
        print("Ultralytics:", ultralytics.__version__)
    except Exception as e:
        print("Ultralytics import error:", e)

    print("\n===== CUDA / GPU =====")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("Device 0:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)

if __name__ == "__main__":
    main()