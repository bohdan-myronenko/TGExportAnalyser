import torch

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("CUDA available:", torch.cuda.get_device_name(0))
    else:
        raise SystemExit("CUDA not available – check your PyTorch/CUDA installation")
