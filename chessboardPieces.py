import cv2
import numpy as np
import random
import copy

def image_processing2(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe_img.apply(gray_img)

    gaussian_blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)
    
    return gaussian_blurred

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
    


    return best_squares


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
def drawSquares(square_matrix, img,piece_presence=None):
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
            color = (0, 255, 0)
            if piece_presence is not None:
                if piece_presence[row_idx][column_idx] == 0:
                    text = "-P"  #Not a piece
                    color = (255, 0, 0)
                else:
                    text = "+P" #Piece present
                    color = (0, 0, 255)
                font_scale = 1
                thickness=2
            else:
                text = f"R{row_idx},C{column_idx}"
                font_scale = 0.5
                thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Adjust text position to center it properly
            text_x = cx - text_size[0] // 2
            text_y = cy + text_size[1] // 2
            
            # Draw the text (white color)
            cv2.putText(
                img, text, (text_x, text_y), 
                font, font_scale, color, thickness, cv2.LINE_AA
            )
            #print(square[0][0][0] -square[2][0][0])
            #print(square[1][0][0] -square[3][0][0])
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
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
        currentLevel = dealWithTooManySquares(currentLevel) # test this
    currentLevel =map(lambda x: x[2],currentLevel)
    return currentLevel
def dealWithTooManySquares(currentLevel):
    while len(currentLevel) > 8:
        smallestDistance = 1000000
        to_remove = None
        for i in range(len(currentLevel)-1):
            if abs(currentLevel[i][1] - currentLevel[i+1][1]) < smallestDistance:
                smallestDistance = abs(currentLevel[i][1] - currentLevel[i+1][1])
                to_remove = i
        currentLevel.pop(to_remove)
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
    if percentage > 30:
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
    #canny_edges = cv2.erode(canny_edges, None, iterations=4)
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

def getNumberOfSquaresNotNull(square_matrix):
    number_of_squares_detected = 0
    cloned = copy.deepcopy(square_matrix)
    for row in cloned:
        for square in row:
            if square is not None:
                number_of_squares_detected += 1
    return number_of_squares_detected
