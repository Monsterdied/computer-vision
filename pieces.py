# draw the bounding boxes of the pieces
import cv2
import numpy as np
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
    #combined = cv2.bitwise_or(edges, thresh)
    #combined = thresh
    best_bounding_boxes = None
    best_processed = None
    best_n_bounding_boxes = -1
    for i in range(2):
        processed,bounding_boxes =optimizeBoundingBoxes(i,thresh,edges)
        if len(bounding_boxes) > best_n_bounding_boxes:
            best_n_bounding_boxes = len(bounding_boxes)
            best_bounding_boxes = bounding_boxes
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
    return bounding_boxes

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
        area = w*h
        aspect_ratio = float(w)/h
        # Filter based on reasonable chess piece characteristics
        if (area < 4000 or area > 70000 or aspect_ratio < 0.3 or aspect_ratio > 3.0):  # Aspect ratio constraints
            #print("bad")
            continue

        #print(f"Area: {area}")
        bounding_boxes.append((x, y, w, h))
    return processed,bounding_boxes
