import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def orb_fun():
    orb = cv2.ORB_create() 

    kp1, des1 = orb.detectAndCompute(query, None)
    kp2, des2 = orb.detectAndCompute(train, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    match_output = cv2.drawMatches(query, kp1, train, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)   

    # Resize match output for visualization
    match_output_resized = cv2.resize(match_output, (0, 0), fx=0.25, fy=0.25)

    cv2.imshow("Matches (Resized)", match_output_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sift_fun(trainpath):

    train = cv2.imread(trainpath, cv2.IMREAD_GRAYSCALE)


    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(query, None)
    kp2, des2 = sift.detectAndCompute(train, None)

    # Create a Brute Force Matcher object
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # Match descriptors using KNN
    knn_matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:  # Adjusted ratio for better results
            good_matches.append(m)

    # Draw matches
    match_output = cv2.drawMatches(
        query, kp1, 
        train, kp2, 
        good_matches, 
        None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    match_output_resized = cv2.resize(match_output, (0, 0), fx=0.25, fy=0.25)
    cv2.imshow("SIFT Matches " + trainpath, match_output_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

query = cv2.imread("assets/board2.jpg",cv2.IMREAD_GRAYSCALE)

for filename in os.listdir("images"):
        trainpath = os.path.join("images", filename)
        print(f"Processing {trainpath}...")
        sift_fun(trainpath)

sift_fun()
# orb_fun()

