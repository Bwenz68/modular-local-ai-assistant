"""
Entry-point stub for the local AI Assistant container.
For now, it just confirms the container boots and CUDA is visible.
"""

import torch
from datetime import datetime

def main():
    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU only"
    print(f"[{datetime.now().isoformat(timespec='seconds')}] "
          f"Assistant container started · CUDA: {cuda_ok} · Device: {gpu_name}")

if __name__ == "__main__":
    main()
