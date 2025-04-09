from chessboard import detect_chessboard, wrap_chessboard
from chessboardPieces import check_pieces,  detect_chessboard_squares, drawSquares
from pieces import get_pieces_bounding_boxes, draw_bounding_boxes
import cv2
import os
import copy
dataDir = "images/" 
count=0
total=0
square_box = None
cannyEdges = None
wrap = None
presence_matrix= None
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
        presence_matrix,total_pieces = check_pieces(copy.deepcopy(square_box),normalizedBoard)
        newImage = cv2.cvtColor(normalizedBoard, cv2.COLOR_GRAY2BGR)
        #drawSquares(square_box,newImage,piece_presence=presence_matrix)
        print("Pieces detected",total_pieces)
        for row in presence_matrix:
            print(row)

    if normalizedBoard is not None:
        # Get bounding boxes of pieces
        bounding_boxes = get_pieces_bounding_boxes(normalizedBoard,True)
        # Draw bounding boxes on the original image
        #convert to color image for visualization
        #normalizedBoard1 = cv2.cvtColor(normalizedBoard, cv2.COLOR_GRAY2BGR)
        #img_with_boxes = draw_bounding_boxes(normalizedBoard1, bounding_boxes)
        #cv2.imshow("Bounding Boxes", img_with_boxes)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    else:
        print("No canny edges found")
    #break
    if presence_matrix is not None:
        presence_matrix = [[0]*8]*8
    # TODO CHECK IF ALL THE COLUMNS ARE DETECTED ADD FILL 
    if len(presence_matrix) != 8:
        rest = [[0]*8]*(8-len(presence_matrix))
        presence_matrix.extend(rest)
    

print(f"Chessboard found in {count} out of {total} images")