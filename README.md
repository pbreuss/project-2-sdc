## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./output_images/final.jpg)

In this project, the goal was to write a software pipeline to identify the lane boundaries in a video, but the main output or product was to create is a detailed writeup of the project. 

## Summary - The goals / steps of this project were the following:

* STEP 1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* STEP 2: Apply a distortion correction to raw images.
* STEP 3: Use color transforms, gradients, etc., to create a thresholded binary image.
* STEP 4: Apply a perspective transform to rectify binary image ("birds-eye view").
* STEP 5: Detect lane pixels and fit to find the lane boundary.
* STEP 6: Determine the curvature of the lane and vehicle position with respect to center.
* STEP 7: Warp the detected lane boundaries back onto the original image.
* STEP 8: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration were stored in the folder called `camera_cal`. The images in `test_images` are for testing the pipeline on single frames. 

## Now the details:

# STEP 1: (camera_calibration.py)
The camera calibration was done in a separate python script (camera_calibration.py), because it is usually run only once. The code has been taken more or less from Lession 6 "Camera Calibration". To calibrate a camera, an image with known proportion can be taken, like a chess board. In this case a 9x6 chessboard was photographed from many angles and perspective. Then each image is read and converted to gray, and the OpeCV function cv2.findChessboardCorners() is called. This function returns a list of 2d image points / coordinates of the chessboard tiles. This list is appended to a list of imagepoints, while a corresponding list of 3d object points is written to a list called objectpoints. Object points, is a prodefined list like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0). After all images have been read, the opencv function cv2.calibrateCamera is used to calculate calibration matrix and distortion coefficients. At leas 20 images are needed for this calibration process.

At the end of this script, the newly calculated calibration matrix and distortion coefficients can be tested on an undistored image.

After undistortion, this image ![calibration1.jpg](./camera_cal/calibration1.jpg) 


becomes this image: ![calibration1_undistorted.jpg](./output_images/calibration1_undistorted.jpg)

These are the calculated calibration matrix and distortion coefficients which are going to be hardcoded in part 2 of this submission (lane_detection.py).

```
Distortion Coefficient:
[[-0.24688775 -0.02373133 -0.00109842  0.00035108 -0.00258571]]
Camera Matrix:
[[1.15777930e+03 0.00000000e+00 6.67111054e+02]
 [0.00000000e+00 1.15282291e+03 3.86128937e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
```


lane_line_detection.py is the main program for this project. In this program we open a video (or simgle image) and apply the "pipeline" to each frame. STEP 2-8 are in this python script.

For each frame the first thing we do is to undistort the frame, according to the calibration matrix and distortion coefficients from STEP 1. We hard coded these values in this python script. Line 21, 22:

```
dist = np.matrix([-0.24688775,-0.02373133,-0.00109842,0.00035108,-0.00258571])
mtx = np.matrix([[1.15777930e+03,0.00000000e+00,6.67111054e+02],[0.00000000e+00,1.15282291e+03,3.86128937e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]])
```

# STEP 2: Apply a distortion correction to raw images (lane_line_detection.py)

In Line 231 we undistort the frame with the opencv function undistort and the calibration matrix and distortion coefficients from the previous step.
```
undistortedFrame = cv2.undistort(originalFrame, mtx, dist, None, mtx)
```

# STEP 3 - Use color transforms, gradients, etc., to create a thresholded binary image

In the function get_warped_binary we do all the steps recommended in Lesson 8, Gradients and Color spaces. This includes converting the image to HLS color space and using Sobel on the L channel in the x direction (x-gradient). Additionally we absolute the x derivative to accentuate lines away from horizontal. Finally we normalize sobelx values and we create a binary image by using some treshholds. Note: to view your binary image use ```  cv2.imshow('sxbinary', sxbinary*255) ``` (multiplay the image by 255, otherwise its black!)

```
# Convert to HLS color space and separate the V channel
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
l_channel = hls[:,:,1]
s_channel = hls[:,:,2]

# Sobel x
sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

# Threshold x gradient
sxbinary = np.zeros_like(scaled_sobel)
sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
```

The resulting image looks like this (x-gradient): ![sxbinary.jpg](./output_images/sxbinary.jpg)

After this we treshhold the s_channel (as recommended in Lessom 8)

```
# Threshold color channel
s_binary = np.zeros_like(s_channel)
s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

```

The resulting image looks like this (s_channel): ![s_binary.jpg](./output_images/s_binary.jpg)

After this step we combine both binary images (x-gradient and s_channel)

```
# Combine the two binary thresholds
combined_binary = np.zeros_like(sxbinary)
combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1    
```

Here is the result:
![combined_binary.jpg](./output_images/combined_binary.jpg)

Next, we apply a mask, to get rid of parts of the image that might give wrong clues about lane lines. Here is the mask I use:

![mask.jpg](./output_images/mask.jpg)

Note, the mask is applied to the binary image, not to the original image!


# STEP 4: Apply a perspective transform to rectify binary image ("birds-eye view").

```
binary_warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)
```

The Matrix M (and Minv) has been created previouly, according to the concept of "Perspective Transform" as explained in Lesson 7. By specifying 2d coordinates of your 3d image, and transforming them to a square, you get a birds eye view of your street.

```
# source and destination points (top left, top right, bottom right, bottom left) for warping street to birds view 
src = np.float32([[600, 450], [690, 450], [1100, 680], [280, 680]])     # these are the coodinates of the street

dst = np.float32([[offset_x, offset_y], [img_size[0]-offset_x, offset_y], [img_size[0]-offset_x, img_size[1]-offset_y], [offset_x, img_size[1]-offset_y]])
                
# use cv2.getPerspectiveTransform() to get M and Minv, the transform matrix and inverse transform matrices to warp the street to birds view and back
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst,src)
```

The resulting "warped" image looks like this:

![binary_warped.jpg](./output_images/binary_warped.jpg)

# coming soon
# STEP 5 - Detect lane pixels and fit to find the lane boundary

# STEP 6 - Determine the curvature of the lane and vehicle position with respect to center

# STEP 7 - Warp the blank back to original image space using inverse perspective matrix (Minv)

# STEP 8 Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
