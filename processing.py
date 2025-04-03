import cv2
import numpy as np
import os
from scipy import ndimage
import math
from matplotlib import pyplot as plt


def image_processing(imgpath):

    img =  cv2.imread(imgpath)

    resized_img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    ret, th_otsu_blur = cv2.threshold(blurred_img,200, 255, cv2.THRESH_BINARY)
    masked_image = cv2.bitwise_and(blurred_img, blurred_img, mask=th_otsu_blur)
    #th_otsu_blur = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return masked_image


def detect_chessboard(imgpath):

    #Image for visualization
    visualize = cv2.imread(imgpath)
    
    #Resize for visualization
    resized_img = cv2.resize(visualize, (0,0), fx=0.25, fy=0.25)

    #Preprocessing of the image
    img = image_processing(imgpath)

    #Canny edge detection
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    edges = cv2.dilate(edges, None, iterations=3)
    edges = cv2.erode(edges, None, iterations=3)

    # Find contours in the image
    contours , _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and filter out small ones
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

    # Display the original image, edges, and detected corners
    cv2.imshow("Processed Image", img)
    cv2.imshow("Edges", edges)
    cv2.imshow("Detected Corners", resized_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return corners



def detect_chessboard2(imgpath):
    img = cv2.imread(imgpath)
    if img is None:
        print("Error: Could not read image")
        return None
        
    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (13,13), 0)
    
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)
    
    # Find centroids
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    
    if corners is not None:
        corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
        
        # Visualization (resized for display only)
        vis = img.copy()
        for corner in corners:
            x,y = corner.ravel()
            cv2.circle(vis, (int(x),int(y)), 10, (0,0,255), -1)
        
        # Resize just for display
        display_scale = 0.25  # Adjust as needed
        resized_vis = cv2.resize(vis, (0,0), fx=display_scale, fy=display_scale)
        cv2.imshow('Harris Corners', resized_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Return corners at original scale
        return corners.reshape(-1, 2)
        
    return None



dataDir = "images/" 
count=0
total=0
for img in os.listdir(dataDir):
    total+=1
    imgpath = os.path.join(dataDir, img)
    corners = detect_chessboard(imgpath)
    if corners is not None:
        count+=1
        print(f"Chessboard found in {img}")
    else:
        print(f"No chessboard found in {img}")
    break

print(f"Chessboard found in {count} out of {total} images")



 