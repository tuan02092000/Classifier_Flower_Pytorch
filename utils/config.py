import torch
import os

# define path to the dataset
DATA_PATH = "flower_photos"
BASE_PATH = "dataset"

# define validation split paths to separate train and validation
VAL_SPLIT = 0.1
TRAIN = os.path.join(BASE_PATH, "train")
VAL = os.path.join(BASE_PATH, "val")

# Hyper parameter
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

FEATURE_EXTRACTION_BATCH_SIZE = 64
FINETUNE_BATCH_SIZE = 32
PRED_BATCH_SIZE = 4
EPOCHS = 20
LR = 0.001
LR_FINETUNE = 0.0005

# define paths to store training plots and trained model
WARMUP_PLOT = os.path.join("output", "warmup.png")
FINETUNE_PLOT = os.path.join("output", "finetune.png")
WARMUP_MODEL = os.path.join("output", "warmup_model.pth")
FINETUNE_MODEL = os.path.join("output", "finetune_model.pth")

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"