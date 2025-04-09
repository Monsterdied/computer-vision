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


def detect_chessboard(imgpath,iteration,debug=False):
    probablistic = False
    if iteration > 3:
        iteration = iteration -2
        probablistic = True
    #Image for visualization
    visualize = cv2.imread(imgpath)
    
    #Resize for visualization
    resized_img = cv2.resize(visualize, (0,0), fx=0.25, fy=0.25)

    #Preprocessing of the image
    img = image_processing(imgpath)

    #Canny edge detection
    canny = cv2.Canny(img, 50, 150, apertureSize=3)
    if not probablistic:
        edges = cv2.dilate(canny, None, iterations=8 + iteration)
        edges = cv2.erode(edges, None, iterations=12 + iteration//4)

    #try hought lines
    if probablistic:
        edges = probabilistic_hough_lines(canny, iteration,canny)

    # Find contours in the image
    contours ,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area and filter out small ones
    largest_area = 0
    chessboard_contour = None
    liste = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            liste.append(approx)
            if area > largest_area:
                largest_area = area
                chessboard_contour = approx
    
    if chessboard_contour is None:
        print("No chessboard found")
        return None,None
    
    corners = chessboard_contour.reshape(4, 2)
    
    # Draw corners on the image
    for corner in corners:
        cv2.circle(resized_img, tuple(corner), 5, (0, 255, 0), -1)

    # Draw the contour
    cv2.polylines(resized_img, [chessboard_contour], True, (255, 0, 0), 2)

    # Display the original image, edges, and detected corners
    if debug == True:
        cv2.imshow("Processed Image", img)
        cv2.imshow("Edges", edges)
        cv2.imshow("Detected Corners", resized_img)

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return corners,largest_area

def probabilistic_hough_lines(edges, iteration,canny):
    edges = cv2.dilate(canny, None, iterations=3 + iteration//2)
    edges = cv2.erode(edges, None, iterations=3 + iteration)
    edges = cv2.dilate(edges, None, iterations=3 + iteration//2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100 + 10 *iteration, minLineLength=170, maxLineGap=10)
    linesImg = np.zeros(edges.shape, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(linesImg, (x1, y1), (x2, y2), 255, 2)  # Green line (BGR)
    linesImg = cv2.dilate(linesImg, None, iterations=1+iteration)
    #linesImg = cv2.erode(linesImg, None, iterations=3)
    #cv2.imshow("Hough Lines", edges)
    return linesImg

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

def draw_lines(lines, line_image,debug=False):
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
            x1 = int(x0 + 1200 * (-b))
            y1 = int(y0 + 1200 * (a))
            x2 = int(x0 - 1200 * (-b))
            y2 = int(y0 - 1200 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
    #img2 = cv2.resize(img2, (0,0), fx=0.25, fy=0.25)
    if debug == True:
        cv2.imshow("Hough Lines", line_image)
        cv2.waitKey(0)
    return line_image

def detect_chessboard_squares(img,debug=False):
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

    adaptive_thresh = cv2.adaptiveThreshold(processed_img, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    canny_edges = cv2.bitwise_or(canny_edges, adaptive_thresh)

    #cv2.imshow("Canny", canny_edges)
    #cv2.imshow("Processed Image", processed_img)
    #cv2.waitKey(0)
    #rezize for visualization remove after debug
    num_votes = 850
    best_squares_number = 0
    best_squares = None
    best_img_lines = None
    #try 3 times to find the best number of votes
    for i in range(7):
        lines = cv2.HoughLines(canny_edges, 1, np.pi / 180, num_votes,0,0)
        #lines = cv2.HoughLinesP(canny_edges, 1, np.pi / 180, threshold=num_votes, minLineLength=10, maxLineGap=1000)
        new_img_lines = np.zeros(canny_edges.shape, dtype=np.uint8)
        image_with_lines = draw_lines(lines, new_img_lines)
        image_with_lines = cv2.dilate(image_with_lines, None, iterations=3)
        squares_raw,matrix = squares(image_with_lines,processed_img,debug=False)
        squares_number = len(squares_raw)
        if squares_number > best_squares_number:
            best_squares_number = squares_number
            best_squares = matrix
            best_img_lines = image_with_lines
        if squares_number == 64:
            break
        else:
            num_votes -= 50
            #print(num_votes)
            #print("Not enough squares found")
    if best_squares is not None:
        if best_squares_number == 64:
            print("All squares found")
        #else:
        best_img_lines = cv2.cvtColor(best_img_lines, cv2.COLOR_GRAY2BGR)
        if debug == True:
            drawSquares(best_squares, best_img_lines)
    


    return best_squares,processed_img


#gets the squares from the image
def squares(img,original_img,debug=False):
        # Find contours in the image
    contours , _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and filter out small ones
    squares = []
    for contour in contours:
        #width height ratio
        x,y,w,h = cv2.boundingRect(contour)

        aspect_ratio = float(w)/h
        #if aspect_ratio < 0.5 or aspect_ratio > 1.5:
        #    continue
        area = cv2.contourArea(contour)
        if 4000 < area and area< 11000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                squares.append(approx)
    #convert to color image for visualization
    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    square_matrix = orderSquares(squares)
    if debug == True:
        drawSquares(square_matrix, original_img)
    return squares,square_matrix


#draws the squares on the image
def drawSquares(square_matrix, img):
    row_idx = -1
    for row in square_matrix:
        counter = 0
        row_idx += 1
        column_idx = -1
        for square in row:
            column_idx += 1
            if square is None:
                #print("Square is None")
                continue
            counter += 1
            #if counter < 8:
            #    continue
            #print(square)
            cv2.drawContours(img, [square], 0, (0, 0, 255), 2)
            moments = cv2.moments(square[2])
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                # Fallback: Use mean of all points if moments fail
                cx = int(np.mean(square[:, 0, 0]))
                cy = int(np.mean(square[:, 0, 1]))
            
            # Display row and column numbers (e.g., "R0,C1")
            text = f"R{row_idx},C{column_idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Adjust text position to center it properly
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            
            # Draw the text (white color)
            cv2.putText(
                img, text, (text_x, text_y), 
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
            )
            #print(square[0][0][0] -square[2][0][0])
            #print(square[1][0][0] -square[3][0][0])
    img = cv2.resize(img, (0,0), fx=0.7, fy=0.7)
    cv2.imshow("Squares drawn", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#Order squares based on their coordinates and converts them to a matrix
def orderSquares(squares,debug=False):
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
        if debug == True:
            print("No squares found")
        return []
    sorted_squares_by_y = sorted(squares_with_centers, key=lambda x: (x[0]))
    current_y = None
    # TODO FINETUNE THIS TO GET LEVELS
    margin_y = 20
    margin_x = 600
    #IMPORTANT
    #each square has 100 by 100 pixels
    matrix = []
    currentLevel = []
    for y,x,square in sorted_squares_by_y:
        if current_y is None:
            current_y = y
            current_x = x
        elif abs(current_y - y) > margin_y or (len(currentLevel) >=8 and abs(current_x - x) > margin_x):
            currentLevel = HandleColumn(currentLevel)
            matrix.append(currentLevel)
            currentLevel = []

        currentLevel.append((y,x,square))
        current_x = x
        current_y = y
    currentLevel = HandleColumn(currentLevel)
    matrix.append(currentLevel)
    return matrix

# check if column is valid
def HandleColumn(currentLevel):
    currentLevel.sort(key=lambda x: x[1])
    if len(currentLevel) < 8:
        currentLevel = dealWithMissingSquares(currentLevel)
    if len(currentLevel) > 8:
        print("Too many squares in a row:",len(currentLevel))
    currentLevel =map(lambda x: x[2],currentLevel)
    return currentLevel

# Deal when a column has missing squares find the NONE squares and add them
def dealWithMissingSquares(currentLevel):
    # TODO probably expand to use the two nearest squares to speculate the middle square  it is probably wrong
    # function defective for now
    previousSquare = None
    result = []
    if len(currentLevel) == 0:
        return [None]*8
    if len(currentLevel) < 2:
        rest = [(-1,-1,None)]*(8-len(currentLevel))
        currentLevel.extend(rest)
        return currentLevel
    previousSquare = currentLevel.pop(0)
    result.append(previousSquare)
    currentEntry = currentLevel.pop(0)
    previousX = currentEntry[1]
    while len(result) <8:
        if abs(currentEntry[1] - previousX) > 120:
            previousX = previousSquare[1] + 100
            #add the missing square to the list
            result.append((currentEntry[0],previousSquare[1],None))
            continue
        result.append(currentEntry)
        previousSquare = currentEntry
        previousX = currentEntry[1]
        if len(currentLevel) == 0:
            rest = [(-1,-1,None)]*(8-len(result))
            result.extend(rest)
            break
        else:
            currentEntry = currentLevel.pop(0)
    return result




#check if the square is black or white
def check_square(square,img,debug):
    #cv2.imshow("Canny Edges", img)
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
    #print(f"Percentage of black pixels: {percentage:.2f}%")
    if debug == True:
        cv2.imshow("Masked Image", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if percentage > 20:
        #print("Black square detected")
        return 0
    else:
        #print("White square detected")
        return 1
    


#check the pressence of pieces in the squares
def check_pieces(square_matrix,img,debug=False):
    canny_edges = cv2.Canny(img, 50, 150, apertureSize=3)
    canny_edges = cv2.dilate(canny_edges, None, iterations=3)
    canny_edges = cv2.erode(canny_edges, None, iterations=1)
    canny_edges = cv2.dilate(canny_edges, None, iterations=8)
    canny_edges = cv2.erode(canny_edges, None, iterations=4)
    if debug == True:
        cv2.imshow("Canny Edges", canny_edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    result = []
    total = 0
    if debug == True:
        cv2.imshow("Canny Edges", canny_edges)
    for row in square_matrix:
        new_row = []
        for square in row:
            if square is None:
                #I dont know just try our luck
                #if it didnt detect a square probably there is a piece there
                new_row.append(1)
                continue
            PiecePresence = check_square(square,canny_edges,debug)
            total += PiecePresence
            new_row.append(PiecePresence)
        result.append(new_row)
    return result,total


# draw the bounding boxes of the pieces
def draw_bounding_boxes(img, bounding_boxes):
    for (x, y, w, h) in bounding_boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
    return img


#get the bounding boxes of the pieces
def get_pieces_bounding_boxes(normalizedBoard,debug=False):
    #canny
    # Apply adaptive thresholding before Canny
    blurred = cv2.GaussianBlur(normalizedBoard, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    # Improved Canny edge detection
    sigma = 1
    v = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blurred, lower, upper)


    # Combine edges with threshold
    combined = cv2.bitwise_or(edges, thresh)
    #combined = cv2.addWeighted(edges, 0.5, thresh, 0.5, 0)
    #combined = thresh
    # Better morphological processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    processed = cv2.erode(combined, kernel, iterations=3)
    processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=5)
    #processed = cv2.dilate(processed, None, iterations=5)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w*h
        aspect_ratio = float(w)/h
        # Filter based on reasonable chess piece characteristics
        #if (area < 300 or area > 10000 or aspect_ratio < 0.3 or aspect_ratio > 3.0):  # Aspect ratio constraints
            #print("bad")
            #continue

        #print(f"Area: {area}")
        bounding_boxes.append((x, y, w, h))
    normalizedBoard1 = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    img_with_boxes = draw_bounding_boxes(normalizedBoard1, bounding_boxes)
    if debug:
        cv2.imshow("Bounding Boxes", img_with_boxes)
        cv2.waitKey(0)
    # Draw bounding boxes on the original image
    return bounding_boxes


dataDir = "images/" 
count=0
total=0
square_box = None
cannyEdges = None
wrap = None
for img in os.listdir(dataDir):
    #img = "G028_IMG015.jpg"

    total+=1
    imgpath = os.path.join(dataDir, img)
    #try to find the board
    for i in range(13):
        corners,curr_area = detect_chessboard(imgpath,i,debug=False)
        if corners is not None:
            print(f"Chessboard found in {img}")
        else:
            continue
        # warp the image
        if corners is not None:
            wrap = wrap_chessboard(imgpath, corners)
        # see if the square warped is a board
        if wrap is not None:
            square_box,normalizedBoard = detect_chessboard_squares(wrap,False)
        if square_box is None:
            #cv2.imshow("No squares found", wrap)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            continue
        else:
            if len(square_box) != 8:
                print("Not all columns detected")
                continue
            #print("Squares found1",len(square_box))
            count+=1
            break
    if square_box is None:
        print("No squares found",imgpath)
            

    if square_box is None:
        print("No wrapping performed")
    #check presence of pieces in the squares
    if square_box is not None:
        matrix,total_pieces = check_pieces(square_box,normalizedBoard)
        print("Pieces detected",total_pieces)
        for row in matrix:
            print(row)

    if normalizedBoard is not None:
        # Get bounding boxes of pieces
        bounding_boxes = get_pieces_bounding_boxes(normalizedBoard)
        # Draw bounding boxes on the original image
        #convert to color image for visualization
        normalizedBoard1 = cv2.cvtColor(normalizedBoard, cv2.COLOR_GRAY2BGR)
        img_with_boxes = draw_bounding_boxes(normalizedBoard1, bounding_boxes)
        #cv2.imshow("Bounding Boxes", img_with_boxes)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    else:
        print("No canny edges found")
    #break
    # TODO CHECK IF ALL THE COLUMNS ARE DETECTED ADD FILL 
    if len(matrix) != 8:
        rest = [[0]*8]*(8-len(matrix))
        matrix.extend(rest)
    

print(f"Chessboard found in {count} out of {total} images")



 