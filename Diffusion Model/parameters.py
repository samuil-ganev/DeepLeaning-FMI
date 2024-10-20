import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

batch = 1
lr = 0.001 # 0.0001 - SD
epochs = 100

clip_layers_val = 5 # 12
clip_heads_val = 4 # 12
d_cross_val = 64 # 768

inf_steps = 5
train_steps = 200

unkToken = '<UNK>'
padToken = '<PAD>'
startToken = '<S>'
endToken = '</S>'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

model_file = 'D:\\FMI\\Deep Learning with PyTorch\\Image Generator\\data\\v1-5-pruned-emaonly.ckpt'

tokensDataFile = 'token2ind'
savedModelFileName = 'model'
mse_osdFile = 'mse_osd'