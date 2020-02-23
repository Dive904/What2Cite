import torch
from torchtext import data

SEED = 2019
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
