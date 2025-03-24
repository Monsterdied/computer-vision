import cv2
import numpy as np
import os
from scipy import ndimage
import math
from matplotlib import pyplot as plt

#include <opencv2/calib3d.hpp>

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

def detect_chessboard(imgpath):
    
    img = cv2.imread(imgpath)

    resized_img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    edges = cv2.Canny(blurred_img, 50, 150, apertureSize=3)

    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=2)

    contours , _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    chessboard_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > largest_area:
                largest_area = area
                chessboard_contour = approx
    
    if chessboard_contour is None:
        print("No chessboard found")
        return None
    
    corners = chessboard_contour.reshape(4, 2)

    # Draw corners on the image
    for corner in corners:
        cv2.circle(resized_img, tuple(corner), 5, (0, 255, 0), -1)

    # Draw the contour
    cv2.polylines(resized_img, [chessboard_contour], True, (255, 0, 0), 2)

    cv2.imshow("Detected Corners", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return corners



dataDir = "images/" 
count=0
total=0
for img in os.listdir(dataDir):
    total+=1
    if img.endswith(".jpg"):
        imgpath = os.path.join(dataDir, img)
        corners = detect_chessboard(imgpath)
        if corners is not None:
            count+=1
            print(f"Chessboard found in {img}")
        else:
            print(f"No chessboard found in {img}")

print(f"Chessboard found in {count} out of {total} images")                                                                     



 