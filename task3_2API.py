from chessboard import detect_chessboard, wrap_chessboard
from chessboardPieces import check_pieces,  detect_chessboard_squares, drawSquares
from pieces import get_pieces_bounding_boxes, draw_bounding_boxes,transform_contours_to_original,draw_bounding_boxes
import copy
def detectBoardAndSquares(imgpath,debug=False):
    for i in range(30):
        corners,curr_area = detect_chessboard(imgpath,i,debug=False)
        if corners is not None:
            if debug:
                print(f"Chessboard found in {imgpath}")
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
                if debug:
                    print("Not all columns detected")
                continue
            #print("Squares found1",len(square_box))
            break
    return normalizedBoard, square_box, M, fx, fy