from chessboard import detect_chessboard, wrap_chessboard,wrapInsideSquare
from chessboardPieces import check_pieces,  detect_chessboard_squares, drawSquares
from pieces import get_pieces_bounding_boxes, draw_bounding_boxes,transform_contours_to_original,draw_bounding_boxes
import cv2
import os
import copy
import json
dataDir = "images/" 
dataDir = os.listdir(dataDir)
result=[]
# check if file input.json exists
if os.path.exists("input.json"):
    with open("input.json", "r") as f:
        json1 = json.load(f)
        dataDir = ""
        images = json1["image_files"]
else:
    dataDir = "images/" 
    images = os.listdir(dataDir)
count=0
total=0
square_box = None
wrap = None
presence_matrix= None
for img in images:
    #img = "G019_IMG082.jpg"

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
            normalizedBoard,M,fx,fy = wrap_chessboard(imgpath, corners)
        # see if the square warped is a board
        #if wrap is not None:
        #    insideSquare = wrapInsideSquare(wrap,False)
        if normalizedBoard is not None:
            square_box = detect_chessboard_squares(normalizedBoard,False)
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
        presence_matrix,total_pieces = check_pieces(copy.deepcopy(square_box),normalizedBoard,False)
        print("Pieces detected",total_pieces)
        for row in presence_matrix:
            print(row)
        #drawSquares(copy.deepcopy(square_box),normalizedBoard,piece_presence=presence_matrix)

    if normalizedBoard is not None and square_box is not None and presence_matrix is not None:
        # Get bounding boxes of pieces
        if square_box is None:
            square_box = []
        bounding_boxes = get_pieces_bounding_boxes(normalizedBoard,square_box,presence_matrix,False)
        #unwrap the bounding boxes to the original image
        img1 = cv2.imread(imgpath)
        unwrapped_bounding_boxes = transform_contours_to_original(bounding_boxes,M,img1.shape,fx,fy)
        #img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
        img1 = draw_bounding_boxes(img1, unwrapped_bounding_boxes)
        img1 = cv2.resize(img1, (0,0), fx=0.2, fy=0.2)
        #cv2.imshow("Bounding Boxes", img1)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #drawSquares(unwrapped_bounding_boxes,img)
        # Draw bounding boxes on the original image
        #convert to color image for visualization
        #normalizedBoard1 = cv2.cvtColor(normalizedBoard, cv2.COLOR_GRAY2BGR)
        #img_with_boxes = draw_bounding_boxes(normalizedBoard1, bounding_boxes)
        #cv2.imshow("Bounding Boxes", img_with_boxes)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    else:
        bounding_boxes = []
    #break
    if presence_matrix is None:
        presence_matrix = [[0]*8]*8
    # TODO CHECK IF ALL THE COLUMNS ARE DETECTED ADD FILL 
    if len(presence_matrix) != 8:
        rest = [[0]*8]*(8-len(presence_matrix))
        presence_matrix.extend(rest)
    entry = {}
    print(img)
    entry["image"] = img
    entry["board"] = presence_matrix
    entry["num_pieces"] = total_pieces
    # convert bounding boxes to a list of lists
    converted_bounding_boxes = []
    for box in bounding_boxes:
        x, y, w, h = cv2.boundingRect(box)
        entry1 = {}
        entry1["xmin"] = x
        entry1["ymin"] = y
        entry1["xmax"] = x + w
        entry1["ymax"] = y + h
        converted_bounding_boxes.append(entry1)
    entry["bounding_boxes"] = converted_bounding_boxes
    result.append(entry)
    
#save the results to a json file
with open("output.json", "w") as f:
    json.dump(result, f)
print(f"Chessboard found in {count} out of {total} images")