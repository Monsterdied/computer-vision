import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def orb_fun(trainpath): 
    
    # Load the image
    train = cv2.imread(trainpath, cv2.IMREAD_GRAYSCALE)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(query, None)
    kp2, des2 = orb.detectAndCompute(train, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography using RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # Draw matches after removing outliers using RANSAC
    matchesMask = mask.ravel().tolist()
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
    match_output = cv2.drawMatches(query, kp1, train, kp2, matches, None, **draw_params)

    # Resize the output for better visibility
    match_output_resized = cv2.resize(match_output, (0, 0), fx=0.25, fy=0.25)
    cv2.imshow("ORB Matches " + trainpath, match_output_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sift_fun(trainpath):

    train = cv2.imread(trainpath, cv2.IMREAD_GRAYSCALE)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(query, None)
    kp2, des2 = sift.detectAndCompute(train, None)

    # Create a FLANN matcher object
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.53 * n.distance:  # Adjusted ratio for better results
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
        orb_fun(trainpath)


