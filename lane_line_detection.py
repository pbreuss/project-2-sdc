# Advanced Lane Finding Project
# The goals / steps of this project are the following:

# 1 Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. (see camera_calibration.py)
# 2 Apply a distortion correction to raw images.
# 3 Use color transforms, gradients, etc., to create a thresholded binary image.
# 4 Apply a perspective transform to rectify binary image ("birds-eye view").
# 5 Detect lane pixels and fit to find the lane boundary.
# 6 Determine the curvature of the lane and vehicle position with respect to center.
# 7 Warp the detected lane boundaries back onto the original image.
# 8 Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# STEP 1 - camera matrix and distortion coefficient where computed in camera_calibration.py
# hard code the camera matrix and distortion coefficient that we computed in the first step
dist = np.matrix([-0.24688775,-0.02373133,-0.00109842,0.00035108,-0.00258571])
mtx = np.matrix([[1.15777930e+03,0.00000000e+00,6.67111054e+02],[0.00000000e+00,1.15282291e+03,3.86128937e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]])

# constants
ym_per_p = 30/720 # meters per pixel in y dimension    # from example
xm_per_p = 3.7/700 # meters per pixel in x dimension   # from example

# perspective transform constants
img_size = (1280, 720)
offset_x = 300
offset_y = 0

# source and destination points (top left, top right, bottom right, bottom left) for warping street to birds view 
src = np.float32([[600, 450], [690, 450], [1100, 680], [280, 680]])     # these are the coodinates of the street

dst = np.float32([[offset_x, offset_y], [img_size[0]-offset_x, offset_y], [img_size[0]-offset_x, img_size[1]-offset_y], [offset_x, img_size[1]-offset_y]])
                
# use cv2.getPerspectiveTransform() to get M and Minv, the transform matrix and inverse transform matrices to warp the street to birds view and back
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst,src)

# left and right masks used for obtaining binary images
left_mask = np.array([[500, 450], [100, 660], [0, 660], [0, 450]])          
right_mask = np.array([[730, 450], [1080, 660], [1280, 660], [1280, 450]])  
center_mask = np.array([[640, 475], [450, 660], [800, 660]])                

# this function 
def find_lane_pixels(binary_warped, plot_it=False):

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # to visualize the histogram uncomment the next 2 lines
    # plt.plot(histogram)
    # plt.show()

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image; numpy.nonzero() returns a tuple of arrays, one for each dimension of arr, containing the indices of the non-zero elements in that dimension.
    nonzero = binary_warped.nonzero()
    #print(nonzero)

    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        if plot_it:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # leftx, lefty are the pixels part of a lane

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, plot_it=False):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, plot_it)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fit_m = np.polyfit(lefty*ym_per_p, leftx*xm_per_p, 2)   # for radius calculation
    right_fit = np.polyfit(righty, rightx, 2)
    right_fit_m = np.polyfit(righty*ym_per_p, rightx*xm_per_p, 2)  # for radius calculation

    if plot_it:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

        plt.imshow(out_img)
        plt.show()

    # STEP 6 Determine the curvature of the lane and vehicle position with respect to center
    # We'll choose the maximum y-value -1, corresponding to the bottom of the image, where we want radius of curvature
    y_eval = binary_warped.shape[0]-1
    left_curverad = ((1 + (2*left_fit_m[0]*y_eval*ym_per_p + left_fit_m[1])**2)**1.5) / np.absolute(2*left_fit_m[0])
    right_curverad = ((1 + (2*right_fit_m[0]*y_eval*ym_per_p + right_fit_m[1])**2)**1.5) / np.absolute(2*right_fit_m[0])

    # This function returns
    # left_fit - Polynomial coefficients of the left curve to draw the curve on the image
    # right_fit - Polynomial coefficients of the right curve to draw the curve on the image
    # left_curverad - the radious of the left curve
    # right_curverad - the radious of the right curve
    # left_fit_m- Polynomial coefficients of the left curve to calculate the offset of the car
    # right_fit_m- Polynomial coefficients of the right curve to calculate the offset of the car
    return left_fit, right_fit, left_curverad, right_curverad, left_fit_m, right_fit_m


# this function turns an undistorted image into a warped binary
def get_warped_binary(img, s_thresh=(170, 255), sx_thresh=(20, 100)):

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

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1    

    # apply masks
    cv2.fillPoly(combined_binary, np.int_([left_mask]), 0)
    cv2.fillPoly(combined_binary, np.int_([right_mask]), 0)
    cv2.fillPoly(combined_binary, np.int_([center_mask]), 0)

    # get the top-down perspective of the binary image
    img_size = combined_binary.shape[1::-1]
    #print(img_size)

    # STEP 4 - Apply a perspective transform to rectify binary image ("birds-eye view")
    binary_warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)

    return binary_warped

# Get the offset of the vehicle from the center of the lane
def get_offcenter(img, left_fit_m, right_fit_m):
    rows, cols = img.shape[:2]
    y0 = rows-1
    
    left = np.polyval(left_fit_m, y0*ym_per_p)
    right = np.polyval(right_fit_m, y0*ym_per_p)
    center = cols / 2 * xm_per_p
    return center - (left + right)/2    


# custom pipeline to detect öane lines
def pipeline(originalFrame):

    # STEP 2 - first distort the image
    undistortedFrame = cv2.undistort(originalFrame, mtx, dist, None, mtx)

    # STEP 3 - Use color transforms, gradients, etc., to create a thresholded binary image
    # then generate the warped binary - these are the road lanes only from birds view
    binary_warped = get_warped_binary(undistortedFrame)

    # to show uncomment
    #cv2.imshow('result', binary_warped*255)

    # STEP 5 - Detect lane pixels and fit to find the lane boundary
    left_fit_p, right_fit_p, left_curverad, right_curverad, left_fit_m, right_fit_m = fit_polynomial(binary_warped)

    # Create an image to draw the lines on
    rows, cols = binary_warped.shape[:2]
    warp_zero = np.zeros(undistortedFrame.shape[:2], dtype=np.uint8)
    lane_image = np.dstack((warp_zero, warp_zero, warp_zero))

    # generate the plot points
    plot_y = np.linspace(0, rows-1, rows) # return evenly spaced numbers over a specified interval.
    left_fit_x = np.polyval(left_fit_p, plot_y)  # calculate the points for the left lane 
    right_fit_x = np.polyval(right_fit_p, plot_y) # calculate the points for the right lane 

    # Put left and right points together
    leftPoints2Lists = np.vstack([left_fit_x, plot_y])
    rigthPoints2Lists = np.vstack([right_fit_x, plot_y])

    # make array with [x,y],[x,y],... 
    leftPoints = np.transpose(leftPoints2Lists)
    rightPoints = np.flipud(np.transpose(rigthPoints2Lists))
    
    # lets put the points in yet another array 
    leftPointsArray = np.array([leftPoints])
    rightPointsArray = np.array([rightPoints])

    # stack arrays in sequence horizontally (column wise).
    polygon_pts = np.hstack((leftPointsArray, rightPointsArray))

    # draw the polygon/lane onto the warped blank image
    cv2.fillPoly(lane_image, np.int_([polygon_pts]), (0,240, 0))

    # if you want to view the polygon/lane uncomment this
    #cv2.imshow('lane_image', lane_image)
    #cv2.imwrite('output_images/lane_image_with_lane.jpg', lane_image)
    
    # STEP 7 warp back on image
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    lane_image_warped = cv2.warpPerspective(lane_image, Minv, undistortedFrame.shape[1::-1])   # was img before

    # STEP 8 Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    # Combine the result with the original image
    result = cv2.addWeighted(undistortedFrame, 1, lane_image_warped, 0.25, 0)

    # put text on the image
    offcenter = get_offcenter(result, left_fit_m, right_fit_m)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, 'Radius of curvature: {0:>10.3f} m'.format(left_curverad), (20,60), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, 'Distance from lane center: {0:>10.3f} m'.format(offcenter), (20,130), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    return result



# comment this code if you want to visualize the mask ###################################### (End of STEP 3)
'''
originalFrame = cv2.imread("test_images/straight_lines1.jpg")

# apply masks
cv2.fillPoly(originalFrame, np.int_([left_mask]), 0)
cv2.fillPoly(originalFrame, np.int_([right_mask]), 0)
cv2.fillPoly(originalFrame, np.int_([center_mask]), 0)

cv2.imshow('result', originalFrame)
cv2.waitKey(5000)
quit()
'''
######################################

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
#cap = cv2.VideoCapture('challenge_video.mp4')
cap = cv2.VideoCapture('project_video.mp4')


# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
 
# create video output
#out = cv2.VideoWriter('output_images/final.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640, 480))
 
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
 
        result = pipeline(frame)

        # Display the resulting frame
        cv2.imshow('Frame', result)
        #out.write(result)

        # Press Q on keyboard to  exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
 
    # Break the loop
    else: 
        break
 
# When everything done, release the video capture object
cap.release()
#out.release()
 
# Closes all the frames
cv2.destroyAllWindows()

'''
# code to test with just one frame
#originalFrame = cv2.imread("test_images/straight_lines1.jpg")
#originalFrame = mpimg.imread("test_images/straight_lines1.jpg")
#originalFrame = mpimg.imread("test_images/test2.jpg")
originalFrame = cv2.imread("test_images/test2.jpg")

result = pipeline(originalFrame)

cv2.imshow('result', result)
#cv2.imwrite('output_images/final.jpg', result)
cv2.waitKey(5000)
'''


