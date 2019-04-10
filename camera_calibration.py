# This program is the first step of project 2: Advanced Lane Finding
# STEP 1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

#print(objp)
#quit()

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #cv2.imshow('img',img)
        #cv2.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

testimg = cv2.imread('camera_cal/calibration1.jpg')
cv2.imshow('testimg distorted',testimg)
dst = cv2.undistort(testimg, mtx, dist, None, mtx)

print("Distortion Coefficient: ")
print(dist)
print("Camera Matrix: ")
print(mtx)

# Note: later in the code, you can use the hardcoded distortion coefficient and camera matrix

cv2.imshow('testimg undistorted', dst)
cv2.waitKey(5000)
cv2.destroyAllWindows()