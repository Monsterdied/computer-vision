import cv2
import numpy as np
from chessboardPieces import draw_lines
import copy
def image_processing(imgpath):

    img =  cv2.imread(imgpath)

    resized_img = cv2.resize(img, (0,0), fx=0.25, fy=0.25)

    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    return blurred_img

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
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
    firstPoint = copy.deepcopy(corners[3])
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
    
    # Display the result
    """cv2.imshow("Warped Chessboard", warped_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()"""
    x,y,_ = warped.shape
    fx = (1000/x)
    fy = (1000/y)
    warped = cv2.resize(warped, (0,0), fx=fy, fy=fx)
    first_point_original = firstPoint
    first_point_reshaped = np.array([[[first_point_original[0], first_point_original[1]]]], dtype=np.float32)
    first_point_warped = cv2.perspectiveTransform(first_point_reshaped, M)[0][0]
    #print("Test1",first_point_warped)
    x = round(first_point_warped[0])
    y = round(first_point_warped[1])
    rotation = 90
    if x < 10 and y > 10:
        rotation = 180
    if x > 10 and y < 10:
        rotation = 0
    if x > 10 and y > 10:
        rotation = 270
    

    #print("Test2",first_point_resized)
    # 6. Draw the first point on the warped image (red circle)
    test = (0,0)
    first_point_warped = [int(first_point_warped[0] * fy), int(first_point_warped[1] * fx)]
    if first_point_warped[0] == 0:
        first_point_warped[0] = 20
    else:
        first_point_warped[0] -= 20
    if first_point_warped[1] == 0:
        first_point_warped[1] = 20
    else:
        first_point_warped[1] -= 20
    """print(fx,fy)
    print(first_point_warped)
    print("IMage shape",warped.shape)
    cv2.circle(warped, (first_point_warped[0],first_point_warped[1]), 100, (0, 0, 255), -1)
    cv2.imshow("Test",warped)
    cv2.waitKey()
    cv2.destroyAllWindows()"""
    return warped,M,fx,fy,rotation

