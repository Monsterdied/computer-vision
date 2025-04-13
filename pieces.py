# draw the bounding boxes of the pieces
import cv2
import numpy as np
from chessboardPieces import drawSquares
def draw_bounding_boxes(img, bounding_boxes):
    for square in bounding_boxes:
        cv2.drawContours(img, [square], -1, (0, 255, 0), 2)  # Draw bounding box in green
    return img


def compute_iou(contour1, contour2):
    """Compute Intersection over Union (IoU) between two contours."""
    # Create blank images
    img1 = np.zeros((1000, 1000), dtype=np.uint8)  # Adjust size if needed
    img2 = np.zeros((1000, 1000), dtype=np.uint8)
    
    # Draw contours
    cv2.drawContours(img1, [contour1], -1, 255, -1)  # Filled contour
    cv2.drawContours(img2, [contour2], -1, 255, -1)
    
    # Compute intersection and union
    intersection = np.logical_and(img1, img2)
    union = np.logical_or(img1, img2)
    
    iou = np.sum(intersection) / np.sum(union)
    return iou


def match_contours_by_iou(contours1, contours2, iou_threshold=0.8):
    """Match contours from list1 to list2 based on IoU."""
    matched_pairs = []
    remaining_contours2 = contours2.copy()
    n_matches = 0
    for contour1 in contours1:
        best_iou = -1
        best_match = None
        best_idx = -1
        # Find the best match in contours2
        for idx, contour2 in enumerate(remaining_contours2):
            iou = compute_iou(contour1, contour2)
            if iou > best_iou:
                best_iou = iou
                best_match = contour2
                best_idx = idx
        
        # If a good match was found, add the pair and remove from remaining_contours2
        if best_iou >= iou_threshold:
            matched_pairs.append( best_match)
            n_matches += 1
            remaining_contours2.pop(best_idx)
        else:
            # If no match, keep contour1 as-is
            matched_pairs.append(contour1)
    
    return matched_pairs, n_matches

def get_ocupied_squares(squares_bb,presence_matrix):
    # Get the bounding boxes of the squares that are occupied by pieces
    occupied_squares = []
    squares_bb = list(map(list,squares_bb))
    for i, row in enumerate(presence_matrix):
        for j, presence in enumerate(row):
            if presence==1 and squares_bb[i][j] is not None:
                occupied_squares.append(list(squares_bb[i])[j])
    return occupied_squares

#get the bounding boxes of the pieces
def get_pieces_bounding_boxes(normalizedBoard,squares,ocupancyMatrix,debug=False):
    ocupiedSquares = get_ocupied_squares(squares,ocupancyMatrix)
    generatedBoundingBoxes = generate_new_bounding_boxes(normalizedBoard,ocupiedSquares,debug)

    #conver to bgr
    #test1 = cv2.cvtColor(normalizedBoard, cv2.COLOR_GRAY2BGR)
    #test = draw_bounding_boxes(test1, generatedBoundingBoxes)
    #cv2.imshow("Matched Bounding Boxes", test)
    #cv2.waitKey(0)
    return generatedBoundingBoxes

def generate_new_bounding_boxes(normalizedBoard,ocupiedSquares, debug):
        #canny
    # conver to black and white
    normalizedBoard = cv2.cvtColor(normalizedBoard, cv2.COLOR_BGR2GRAY)
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
    #combined = cv2.bitwise_or(edges, thresh)
    #combined = thresh
    best_bounding_boxes = None
    best_processed = None
    best_n_bounding_boxes = -1
    for i in range(2):
        processed,bounding_boxes =optimizeBoundingBoxes(i,thresh,edges)
        contours, n_matched = match_contours_by_iou(ocupiedSquares, bounding_boxes, iou_threshold=0.3)
        if len(bounding_boxes) > best_n_bounding_boxes:
            best_n_bounding_boxes = n_matched
            best_bounding_boxes = contours
            best_processed = processed
        print(f"Iteration {i}: Found {len(bounding_boxes)} bounding boxes")


    normalizedBoard1 = cv2.cvtColor(best_processed, cv2.COLOR_GRAY2BGR)
    img_with_boxes = draw_bounding_boxes(normalizedBoard1, best_bounding_boxes)
    normalizedBoard2 = cv2.cvtColor(normalizedBoard, cv2.COLOR_GRAY2BGR)
    ground_truth = draw_bounding_boxes(normalizedBoard2,best_bounding_boxes)
    if debug:
        cv2.imshow("Bounding Boxes", img_with_boxes)
        cv2.imshow("Ground Truth", ground_truth)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # Draw bounding boxes on the original image
    return best_bounding_boxes

def bbox_to_contour(x, y, w, h):
    """Converts a bounding box (x, y, w, h) into a contour (4-point polygon)."""
    contour = np.array([
        [[x, y]],          # Top-left
        [[x + w, y]],      # Top-right
        [[x + w, y + h]],  # Bottom-right
        [[x, y + h]]       # Bottom-left
    ], dtype=np.int32)
    return contour

def optimizeBoundingBoxes(iteration,thresh,edges):
    #kernel =
    edges = cv2.dilate(edges, None, iterations=3 + iteration//2)
    img2_inv = cv2.bitwise_not(edges)
        # AND operation with inverted image
    combined = cv2.bitwise_and(thresh, img2_inv)
    #combined = cv2.addWeighted(edges, 0.5, thresh, 0.5, 0)
    #combined = thresh
    # Better morphological processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    processed = cv2.erode(combined, kernel, iterations=2+ iteration//2)
    processed = cv2.dilate(processed, None, iterations=13+iteration)
    #processed = cv2.erode(processed, kernel, iterations=15)
    #processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=5)
    #

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        area = w*h
        aspect_ratio = float(w)/h
        # Filter based on reasonable chess piece characteristics
        if (area < 4000 or area > 30000 or aspect_ratio < 0.3 or aspect_ratio > 3.0):  # Aspect ratio constraints
            #print("bad")
            continue

        #print(f"Area: {area}")
        bounding_boxes.append(bbox_to_contour(x, y, w, h))
    return processed,bounding_boxes

def transform_contours_to_original(contours_warped, M, original_img_shape,fx,fy):
    """Transforms contours from warped space to original image space.
    Args:
        contours_warped: Contours detected in the warped image
        M: Perspective transform matrix from wrap_chessboard()
        original_img_shape: Shape of the original image (h, w, c)
    Returns:
        contours_original: Contours mapped to original image coordinates
    """
    M_inv = np.linalg.inv(M)  # Inverse perspective transform
    
    contours_original = []
    height, width = original_img_shape[:2]  # Get just height and width
    height = height
    width = width 
    print("original shape",original_img_shape)
    for cnt in contours_warped:
        # Reshape contour to (N, 1, 2) and convert to float32
        cnt_scaled = cnt.copy().astype(np.float32)
        cnt_scaled[..., 0] *= 1/fy  # x-coordinates
        cnt_scaled[..., 1] *= 1/fx  # y-coordinates
        
        # Step 2: Reshape for perspective transform
        cnt_reshaped = cnt_scaled.reshape(-1, 1, 2)
        
        # Step 3: Map to original image space
        cnt_original = cv2.perspectiveTransform(cnt_reshaped, M_inv)
        
        # Clip to image boundaries (separately for x and y coordinates)
        cnt_original[..., 0] = np.clip(cnt_original[..., 0], 0, width-1)   # x coordinates
        cnt_original[..., 1] = np.clip(cnt_original[..., 1], 0, height-1)  # y coordinates
        
        contours_original.append(cnt_original.astype(np.int32))
    
    return contours_original
