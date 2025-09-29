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

    pe = torch.zeros(len(timestamps), TIME_DIM, device=dev, dtype=dtype)
    k2 = torch.arange(0, TIME_DIM // 2, dtype=dtype, device=dev)

    div_term = torch.exp(-k2 / TIME_DIM * math.log(D))

    ts = timestamps.unsqueeze(1).to(device=dev, dtype=dtype)

    pe[:, 0::2] = torch.sin(ts * div_term)
    pe[:, 1::2] = torch.cos(ts * div_term)
    
    return pe
