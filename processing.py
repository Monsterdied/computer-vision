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
    return blurred_img


def image_processing2(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe_img.apply(gray_img)

    gaussian_blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    
    return gaussian_blurred


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


def wrap_chessboard(imgpath, corners):
    if corners is None:
        print("No corners found, skipping wrapping")
        return None
    
    # Load the original image (not resized)
    img = cv2.imread(imgpath)
    
    # Since corners were detected on a resized image, we need to scale them back up
    scale_factor = 4 
    corners = corners * scale_factor
    
    # Order the corners: top-left, top-right, bottom-right, bottom-left
    # First, sort by y-coordinate to separate top and bottom rows
    corners = corners[corners[:, 1].argsort()]
    # Then sort top and bottom rows by x-coordinate
    top_row = corners[:2][corners[:2, 0].argsort()]
    bottom_row = corners[2:][corners[2:, 0].argsort()]
    ordered_corners = np.array([top_row[0], top_row[1], bottom_row[1], bottom_row[0]], dtype=np.float32)
    
    # Calculate the width and height of the chessboard
    width = max(
        np.linalg.norm(ordered_corners[0] - ordered_corners[1]),
        np.linalg.norm(ordered_corners[2] - ordered_corners[3])
    )
    height = max(
        np.linalg.norm(ordered_corners[0] - ordered_corners[3]),
        np.linalg.norm(ordered_corners[1] - ordered_corners[2])
    )
    
    # Create destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(ordered_corners, dst)
    
    # Warp the image
    warped = cv2.warpPerspective(img, M, (int(width), int(height)))

    warped_resized = cv2.resize(warped, (0,0), fx=0.25, fy=0.25)
    
    # Display the result
    cv2.imshow("Warped Chessboard", warped_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return warped

def draw_lines(lines, line_image):
    if lines is not None:
        for line in lines:
            # dont draw diagonal lines
            rho, theta = line[0]
            # Convert theta to degrees for easier understanding
            theta_deg = np.degrees(theta)
            
            # Define margin for horizontal/vertical acceptance (in degrees)
            margin = 10  # degrees of tolerance
            
            # Check if line is approximately horizontal (0° or 180° ± margin)
            is_horizontal = (abs(theta_deg) < margin) or (abs(theta_deg - 180) < margin)
            
            # Check if line is approximately vertical (90° or 270° ± margin)
            is_vertical = (abs(theta_deg - 90) < margin) or (abs(theta_deg - 270) < margin)
            
            # Skip if neither horizontal nor vertical within margin
            if not (is_horizontal or is_vertical):
                continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
    #img2 = cv2.resize(img2, (0,0), fx=0.25, fy=0.25)
    cv2.imshow("Hough Lines", line_image)
    cv2.waitKey(0)

def detect_chessboard_squares(img):
    cv2.imshow("Original Image1", img)
    processed_img = image_processing2(img)
    x,y = processed_img.shape
    fx = (1000/x)
    fy = (1000/y)
    print(fx,fy)
    processed_img = cv2.resize(processed_img, (0,0), fx=fy, fy=fx)
    cv2.imshow("Processed Image1", processed_img)
    canny_edges = cv2.Canny(processed_img, 50, 150, apertureSize=3)
    canny_edges = cv2.dilate(canny_edges, None, iterations=4)
    canny_edges = cv2.erode(canny_edges, None, iterations=3)
    print(x,y)
    print(canny_edges.shape)
    cv2.imshow("Canny", canny_edges)
    cv2.waitKey(0)
    #rezize for visualization remove after debug
    num_votes = 580
    lines = cv2.HoughLines(canny_edges, 1, np.pi / 180, num_votes,0,0)
    #lines = cv2.HoughLinesP(canny_edges, 1, np.pi / 180, threshold=num_votes, minLineLength=10, maxLineGap=1000)
    test = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
    draw_lines(lines, test)

    return canny_edges

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

    if corners is not None:
        wrap = wrap_chessboard(imgpath, corners)
    
    if wrap is not None:
        canny = detect_chessboard_squares(wrap)

    if canny is None:
        print("No wrapping performed")
    break

print(f"Chessboard found in {count} out of {total} images")



 