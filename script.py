import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans(k,img):
    reshaped_image = img.reshape((-1,3))
    reshaped_image = np.float32(reshaped_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Apply K-means
    ret, label, center = cv2.kmeans(reshaped_image, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape((img.shape))
    return result

def edgeDetection(img2):
    # Gaussian Blur
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)
    # Apply Canny filter
    img2_canny = cv2.Canny(img2, 50, 150)

    # Create BGR copy of image
    img2_copy = cv2.cvtColor(img2_canny, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Canny Edge Detection", img2_copy)

    #apply hought lines
    num_votes = 160

    lines = cv2.HoughLines(img2_canny, 1, np.pi / 180, num_votes,None, 0, 0)
    """min_line_length = 10  # Minimum length of a line segment
    max_line_gap = 1000    # Maximum allowed gap between line segments
    lines = cv2.HoughLinesP(img2_canny, 1, np.pi / 180, threshold=num_votes,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)"""
    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img2_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("Hough Lines", img2_copy)

def gpt_Idea(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #TODO DIDNT work
    gray = kmeans(3,img)
    chessboard_size = (7, 7)  # Change this based on your chessboard

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    cv2.imshow("Chessboard Corners", gray)
    if ret:
        # Refine the corner locations for better accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw the corners on the image
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Chessboard corners not found.")

if __name__ == "__main__":
    boardTemplate = "assets/board.jpg"
    testImage1 = "images/G000_IMG062.jpg"
    board = cv2.imread(boardTemplate)
    testImage= cv2.imread(testImage1)
    testImage = cv2.resize(testImage, (board.shape[1], board.shape[0]))

    cv2.imshow("Board", board)
    #kmeans(3,testImage)
    #edgeDetection(testImage)
    gpt_Idea(testImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
