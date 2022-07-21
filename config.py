import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 4
SAVE_MODEL = True
DATASET = "./data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.001
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 14, 6, 3]
IMG_CHANNELS = 3
Z_DIM = 512
IN_CHANNELS = 512
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [28, 54, 54, 54, 54, 54, 54, 54, 54]
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 0


