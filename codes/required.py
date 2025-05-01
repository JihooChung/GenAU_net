pip install kagglehub==0.3.6
pip install albumentations

import kagglehub
import numpy as np
import pandas as pd

import os
import cv2
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

BASE_PATH= "/kaggle/input/lgg-mri-segmentation/kaggle_3m"
BASE_LEN = 27
BASE_LEN = 67
END_LEN = 4
END_MASK_LEN = 9
IMG_SIZE = 512

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
