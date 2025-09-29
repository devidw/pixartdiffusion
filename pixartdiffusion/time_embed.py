import torch
import math

# Sinusoidal timestamp embedding

# must be even
TIME_DIM = 30
D = 10000

def get_timestamp_embedding(timestamps):
    # Ensure all tensors are created on the same device and dtype as timestamps
    dev = timestamps.device
    dtype = torch.float

    # ONNX-friendly implementation without ScatterND assignment
    k2 = torch.arange(0, TIME_DIM // 2, dtype=dtype, device=dev)
    div_term = torch.exp(-k2 / TIME_DIM * math.log(D))
    ts = timestamps.unsqueeze(1).to(device=dev, dtype=dtype)

    s = torch.sin(ts * div_term)  # (N, TIME_DIM//2)
    c = torch.cos(ts * div_term)  # (N, TIME_DIM//2)

    # Interleave [s0,c0,s1,c1,...] via stack+reshape
    pe = torch.stack((s, c), dim=-1).reshape(len(timestamps), TIME_DIM)
    return pe
