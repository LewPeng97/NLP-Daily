import os
import torch
from numpy import random
# root_dir = "/search/odin/liuyouyuan/pyproject/data/finished_files"
root_dir = "/home/penglu/LewPeng/GDE/finished_files/"
train_data_path = os.path.join(root_dir, "chunked/train_*")
eval_data_path = os.path.join(root_dir, "test.bin")
decode_data_path = os.path.join(root_dir, "test.bin")
vocab_path = os.path.join(root_dir, "vocab")
#log_root = "./logs/weibo"
log_root = os.path.join(root_dir, "logs_lstm_transformer_encoder/LCSTS")

# Hyperparameters
hidden_dim = 600
emb_dim = 300
d_model = 300
dropout = 0.1
batch_size = 64

# max_enc_steps = 400
max_enc_steps = 200
# max_dec_steps = 100
max_dec_steps = 40
beam_size = 4
# min_dec_steps = 35
min_dec_steps = 20
vocab_size = 5004
# vocab_size = 50_000
lr = 0.15
# adam_lr = 0.0001    # 使用Adam时候的学习率
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0

#是否使用卷积
swish = True


pointer_gen = True
#pointer_gen = False
# is_coverage = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 1200000

lr_coverage=0.15

# 使用GPU相关
use_gpu = True
GPU = "cuda"
USE_CUDA = use_gpu and torch.cuda.is_available()     # 是否使用GPU
NUM_CUDA = torch.cuda.device_count()
DEVICE = torch.device(GPU if USE_CUDA else 'cpu')

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = True
