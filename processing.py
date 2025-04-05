import cv2
import numpy as np
import os
from scipy import ndimage
import math
from matplotlib import pyplot as plt
import random


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
    """cv2.imshow("Processed Image", img)
    cv2.imshow("Edges", edges)
    cv2.imshow("Detected Corners", resized_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
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
    """cv2.imshow("Warped Chessboard", warped_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    
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
    return line_image

def detect_chessboard_squares(img):
    #cv2.imshow("Original Image1", img)
    processed_img = image_processing2(img)
    x,y = processed_img.shape
    fx = (1000/x)
    fy = (1000/y)
    processed_img = cv2.resize(processed_img, (0,0), fx=fy, fy=fx)
    #cv2.imshow("Processed Image1", processed_img)
    canny_edges = cv2.Canny(processed_img, 50, 150, apertureSize=3)
    canny_edges = cv2.dilate(canny_edges, None, iterations=3)
    canny_edges = cv2.erode(canny_edges, None, iterations=1)
    #cv2.imshow("Canny", canny_edges)
    #cv2.imshow("Processed Image", processed_img)
    #cv2.waitKey(0)
    #rezize for visualization remove after debug
    num_votes = 600
    best_squares_number = 0
    best_squares = None
    #try 3 times to find the best number of votes
    for i in range(3):
        lines = cv2.HoughLines(canny_edges, 1, np.pi / 180, num_votes,0,0)
        #lines = cv2.HoughLinesP(canny_edges, 1, np.pi / 180, threshold=num_votes, minLineLength=10, maxLineGap=1000)
        new_img_lines = np.zeros(canny_edges.shape, dtype=np.uint8)
        image_with_lines = draw_lines(lines, new_img_lines)
        squares_raw,matrix = squares(image_with_lines,processed_img)
        squares_number = len(squares_raw)
        print(f"Squares found: {squares_number}")
        if squares_number > best_squares_number:
            best_squares_number = squares_number
            best_squares = matrix
        if squares_number == 64:
            break
        else:
            num_votes -= 50
            #print(num_votes)
            #print("Not enough squares found")
    


    return best_squares,canny_edges


#gets the squares from the image
def squares(img,original_img):
        # Find contours in the image
    contours , _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and filter out small ones
    squares = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5500 < area < 11000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                squares.append(approx)
    #convert to color image for visualization
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    square_matrix = orderSquares(squares)
    drawSquares(square_matrix, original_img)
    return squares,square_matrix


#draws the squares on the image
def drawSquares(square_matrix, img):
    for row in square_matrix:
        #counter = 0
        for square in row:
            if square is None:
                continue
            cv2.drawContours(img, [square], 0, (0, 0, 255), 2)
            #print(square[0][0][0] -square[2][0][0])
            #print(square[1][0][0] -square[3][0][0])
            #counter += 1
            #if counter > 1:
            #    break
    cv2.imshow("Squares drawn", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Order squares based on their coordinates and converts them to a matrix
def orderSquares(squares):
    # Sort squares based on their coordinates
        # Calculate the center of each square for sorting
    squares_with_centers = []
    for square in squares:
        # Calculate centroid (average of all points)
        x_coords = [point[0][0] for point in square]
        y_coords = [point[0][1] for point in square]
        center_x = sum(x_coords) / 4
        center_y = sum(y_coords) / 4
        squares_with_centers.append((center_y, center_x, square))
    if len(squares_with_centers) == 0:
        print("No squares found")
        return []
    sorted_squares_by_y = sorted(squares_with_centers, key=lambda x: (x[0]))
    current_y = None
    # TODO FINETUNE THIS TO GET LEVELS
    margin_y = 40
    smallest_x = sorted(squares_with_centers,key=lambda x: (x[1]))[0][1]
    #IMPORTANT
    #each square has 100 by 100 pixels
    matrix = []
    currentLevel = []
    for y,x,square in sorted_squares_by_y:
        if current_y is None:
            current_y = y
            currentLevel = [None] * 8
        elif abs(current_y - y) > margin_y:
            current_y = y
            matrix.append(currentLevel)
            currentLevel = [None] * 8
        x_level = round((x-smallest_x) // 100)
        if x_level > 7:
            print("X level out of bounds")
            continue

        currentLevel[x_level] = square
    
    matrix.append(currentLevel)
    # Old Approach
    #sort by x and y with the margin applied to the y axis
    #sorted_squares_by_y_x = sorted(matrix, key=lambda x: (x[0],x[1]))
    #print(sorted_squares_by_y_x)
    #sorted_squares = map(lambda x: x[2],sorted_squares_by_y_x)


        
    return matrix
def check_square(square,img):
    cv2.imshow("Canny Edges", img)
    x, y, w, h = cv2.boundingRect(square)
    roi = img[y:y+h, x:x+w]

    mask = np.zeros((h, w), dtype=np.uint8)
    adjusted_contour = square - [x, y]  # Adjust contour coordinates
    cv2.drawContours(mask, [adjusted_contour], -1, 255, thickness=cv2.FILLED)

    # 4. Apply the mask to the cropped region
    masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
    
    # 5. Resize to desired output size (100x100)
    resized = cv2.resize(masked_roi, (100, 100), interpolation=cv2.INTER_LINEAR)

    total_pixels = resized.size
    black_pixels = np.count_nonzero(resized == 0)
    
    # Calculate percentage
    percentage = (black_pixels / total_pixels) * 100.0
    print(f"Percentage of black pixels: {percentage:.2f}%")
    cv2.imshow("Masked Image", resized)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    if percentage > 20:
        print("Black square detected")
        return 0
    else:
        print("White square detected")
        return 1

def check_pieces(square_matrix,cannyEdges):
    cannyEdges = cv2.dilate(cannyEdges, None, iterations=15)
    cannyEdges = cv2.erode(cannyEdges, None, iterations=10)
    result = []
    for row in square_matrix:
        new_row = []
        for square in row:
            if square is None:
                #I dont know just try our luck
                new_row.append(random.randint(0,1))
                continue
            check_square(square,cannyEdges)
        result.append(new_row)


dataDir = "images/" 
count=0
total=0
for img in os.listdir(dataDir):
    #img = "G083_IMG089.jpg"
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
        square_box,cannyEdges = detect_chessboard_squares(wrap)

    if square_box is None:
        print("No wrapping performed")
    if square_box is not None:
        matrix = check_pieces(square_box,cannyEdges)
    #break

print(f"Chessboard found in {count} out of {total} images")



 