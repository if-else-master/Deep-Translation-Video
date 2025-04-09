import torch
print("MPS 可用:", torch.backends.mps.is_available())
print("MPS 已啟用:", torch.backends.mps.is_built())
