import cv2
import numpy as np
import os
from scipy import ndimage
import math
from matplotlib import pyplot as plt

"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torchsummary import summary

"""

dataDir = "images/"
imgs = []

for file in os.listdir(dataDir):
    img = cv2.imread(os.path.join(dataDir, file))
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs.append(img)

cv2.imshow("Image", imgs[0])

cv2.waitKey(0)

cv2.destroyAllWindows()

