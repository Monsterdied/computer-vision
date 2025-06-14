import copy
from chessboard import detect_chessboard, wrap_chessboard
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from chessboardPieces import detect_chessboard_squares,drawSquares
def detectBoardAndSquares(imgpath,model,debug=False):
    #corners,curr_area = detect_chessboard(imgpath,0,debug=False)
    #print(corners)
    corners = detectBoard(imgpath,model)
    corners = resize_contours(corners)
    if corners is not None:
        if debug:
            print(f"Chessboard found in {imgpath}")
    # warp the image
    #print("Ok")
    #print(corners)
    if corners is not None:
        normalizedBoard,M,fx,fy,rotation = wrap_chessboard(imgpath, corners)
        #print(corners)
        #print(transform_point(corners[3],M))
    # see if the square warped is a board
    #if wrap is not None:
    #    insideSquare = wrapInsideSquare(wrap,False)
    if normalizedBoard is not None:
        #square_box = detect_chessboard_squares(normalizedBoard,False)
        #print(square_box)
        height, width = normalizedBoard.shape[:2]
        square_box = get_8x8_grid_contours(height, width)
        #print(square_box)
    else:
        print("Board detection")
    if square_box is not None:
        if len(square_box) != 8:
            if debug:
                print("Not all columns detected")
    #drawSquares(square_box,normalizedBoard)
    #cv2.imshow("test",normalizedBoard)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    return normalizedBoard, square_box, M, fx, fy,rotation
def transform_point(point, M, fx=1.0, fy=1.0):
    # Convert point to homogeneous coordinates
    pt = np.array([[point]], dtype=np.float32)
    
    # Apply perspective transform
    transformed_pt = cv2.perspectiveTransform(pt, M)
    
    # Apply scaling (if needed)
    x_transformed = transformed_pt[0][0][0] * fx
    y_transformed = transformed_pt[0][0][1] * fy
    
    return (int(x_transformed), int(y_transformed))
def resize_contours(contour, scale_factor=0.25):
    return (contour.astype(np.float32) * scale_factor).astype(np.int32)

def detectBoard(imgpath,model):
    result = model.predict(imgpath, imgsz=640, conf=0.6,save=False,verbose=False)
    result = result[0].cpu().keypoints.xy[0]
    contour = np.array([
        result[0],
        result[1],
        result[3],
        result[2]
    ], dtype=np.int32)
    """img = cv2.imread(imgpath)
    img = cv2.drawContours(img,[contour],contourIdx=-1,color=(0, 255, 0),thickness=2)
    cv2.circle(img, tuple(contour[0]), 30, (0,255,0), -1)  # -1 = filled circle
    cv2.circle(img,  tuple(contour[1]), 30, (0,255,255), -1)  # -1 = filled circle
    cv2.circle(img,  tuple(contour[2]), 30, (255,0,0), -1)  # -1 = filled circle
    cv2.circle(img,  tuple(contour[3]), 30, (0,0,255), -1)  # -1 = filled circle


    img = cv2.resize(img,(720,720))
    cv2.imshow("Test",img)
    cv2.waitKey()
    cv2.destroyAllWindows()"""
    return contour

def get_8x8_grid_contours(image_height, image_width):

    # Calculate the width and height of each square
    square_h = image_height // 8
    square_w = image_width // 8
    
    # Initialize an 8x8 matrix for contours
    grid_contours = [[None for _ in range(8)] for _ in range(8)]
    
    for i in range(8):
        for j in range(8):
            # Define the square's bounding coordinates
            x1 = j * square_w
            y1 = i * square_h
            x2 = (j + 1) * square_w
            y2 = (i + 1) * square_h
            
            # Create a contour in OpenCV format (4 points, int32)
            contour = np.array([
                [[x1, y1]],  # Top-left
                [[x2, y1]],  # Top-right
                [[x2, y2]],  # Bottom-right
                [[x1, y2]],  # Bottom-left
            ], dtype=np.int32)
            
            grid_contours[i][j] = contour
    
    return grid_contours

#detectBoard("images/0/G000_IMG000.jpg",model)
#model = YOLO("pose11n.pt") 
#detectBoardAndSquares("images/0/G000_IMG000.jpg",model)