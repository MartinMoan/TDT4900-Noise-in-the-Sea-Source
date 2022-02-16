#!/usr/bin/env python3
import torch

def main():
    print(f"CUDA avalable?: {torch.cuda.is_available()}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device{i}: {torch.cuda.device(i)}")

if __name__ == "__main__":
    main()