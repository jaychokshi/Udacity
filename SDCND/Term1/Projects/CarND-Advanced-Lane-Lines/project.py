
# coding: utf-8

# # Advanced Lane Finding Project
#
# The goals / steps of this project are the following:
#
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
#
# ---
# ## Import stuff

# In[37]:

import numpy as np
import cv2
import pickle
import glob
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')
'''
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# ## Step 1: Perform camera calibration, store it locally

# In[38]:

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        #img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        # plt.imshow(img)


# In[39]:

# Test undistortion on an image
img = cv2.imread('camera_cal/calibration5.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None)

dst = cv2.undistort(img, mtx, dist, None, mtx)


# ## Step 2: Correct the image distortion

# In[40]:

def calc_undistort(image):
    return cv2.undistort(image, mtx, dist, None, mtx)


# In[41]:

# Test checker image
'''
img = cv2.imread('camera_cal/calibration2.jpg')
img_size = (img.shape[1], img.shape[0])

undistorted = calc_undistort(img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
'''


# In[42]:

# Test raw image
'''
img = cv2.imread('test_images/test1.jpg')
img_size = (img.shape[1], img.shape[0])
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
undistorted = calc_undistort(img)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
'''


# ## Step 3: Add color and gradient threshold methods

# In[43]:

def channel_select(img, thresh=(0, 255)):
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary


# In[44]:

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Create a copy and apply the threshold
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return this mask as your binary_output image
    return binary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    mag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(mag) / 255
    mag = (mag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary = np.zeros_like(mag)
    binary[(mag >= mag_thresh[0]) & (mag <= mag_thresh[1])] = 1
    return binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary = np.zeros_like(absgraddir)
    binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary


# ## Step 4: Add perspective transform mechanic

# In[56]:

def get_perspective_points(image, top=70, bottom=370, h_scale=0.65, h_sub=50):
    h, w = image.shape[0:2]
    midpnt = h / 2
    src = np.float32([[(w / 2) - top, h * h_scale],
                      [(w / 2) + top, h * h_scale],
                      [(w / 2) + bottom, h - h_sub],
                      [(w / 2) - bottom, h - h_sub]])

    dst = np.float32([[(w / 2) - midpnt, (h / 2) - midpnt],
                      [(w / 2) + midpnt, (h / 2) - midpnt],
                      [(w / 2) + midpnt, (h / 2) + midpnt],
                      [(w / 2) - midpnt, (h / 2) + midpnt]])
    return src, dst


def get_perspective_transform(image, top=70, bottom=370, h_scale=0.65, h_sub=50):
    h, w = image.shape[0:2]
    src, dst = get_perspective_points(image)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped


# ## Step 5: Alright, so far so good, Add the method to detect lane points

# In[46]:

def extract_lane_edges(image, kernel_size=5):
    # Extract red channel
    red_channel = image[:, :, 0]

    # Equilize the histogram (same as P1)
    equ = cv2.equalizeHist(red_channel)
    red_binary = channel_select(equ, thresh=(250, 255))
    gradx = abs_sobel_thresh(red_channel, orient='x',
                             sobel_kernel=kernel_size, thresh=(30, 255))
    combined = np.zeros_like(red_channel)
    combined[(red_binary == 1) | (gradx == 1)] = 1

    return combined


# ## Step 6: Add sliding window fit polynomial protocol

# In[47]:

def sliding_window_fit_poly(binary_warped):

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean
        # position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    '''
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()
    '''

    return ploty, left_fitx, right_fitx, left_fit, right_fit


def non_sliding_window_fit_poly(binary_warped, left_fit,  right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) +
                                   left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0] * (nonzeroy**2) +
                                   left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) +
                                    right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0] * (nonzeroy**2) +
                                    right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    return ploty, left_fitx, right_fitx, left_fit, right_fit


# ## Step 7: Now lets calculate the lane curvature

# In[48]:

def lane_curvature(ploty, left_fitx, right_fitx):
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 20 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
                           left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
                            right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

    return (left_curverad + right_curverad) / 2


# ## Step 8: Almost there, render the polynomial on the lane now

# In[49]:

def map_lane(image, ploty, left_fitx, right_fitx):

    src, dst = get_perspective_points(image)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image[:, :, 0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp back to original image
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)


# ## Step 9: Run the pipline on each of the images

# In[57]:

g_left_fit = None
g_right_fit = None


def pipeline_img(input_image):
    global g_left_fit
    global g_right_fit

    undistort_img = calc_undistort(input_image)

    image_t = get_perspective_transform(undistort_img)

    img_binary = extract_lane_edges(image_t)

    if g_left_fit is not None:
        ploty, left_fitx, right_fitx, left_fit, right_fit =
            non_sliding_window_fit_poly(img_binary,
                                        g_left_fit, g_right_fit)
    else:
        ploty, left_fitx, right_fitx, left_fit, right_fit =
            sliding_window_fit_poly(img_binary)

    g_left_fit = left_fit
    g_right_fit = right_fit

    output_lane = map_lane(undistort_img, ploty, left_fitx, right_fitx)

    curv = lane_curvature(ploty, left_fitx, right_fitx)
    output_curvature = cv2.putText(output_lane, "Curvature Radius: " + str(
        int(curv)) + "m", (900, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)

    xm_per_pix = 3.7 / 700
    left_lane_pos = left_fitx[len(left_fitx) - 1]
    right_lane_pos = right_fitx[len(right_fitx) - 1]
    road_pos = (((left_lane_pos + right_lane_pos) / 2) - 640) * xm_per_pix
    output_image = cv2.putText(output_lane, "Center Offset: {0:.2f}m".format(
        road_pos), (900, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)

    if len(output_image.shape) == 2:
        return cv2.cvtColor(np.float32(output_image), cv2.COLOR_GRAY2RGB)
    else:
        return output_image


# In[58]:
'''
test_image = mpimg.imread('test_images/test5.jpg')
test_output = pipeline_img(test_image)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(test_image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(test_output)
ax2.set_title('output Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
'''

# ## Time to process the video(s) of the proejct

# In[53]:

project_output_file = "project_output.mp4"
project_video = VideoFileClip("project_video.mp4")

project_output = project_video.fl_image(pipeline_img)
get_ipython().magic('time project_output.write_videofile(project_output_file,\
				audio=False)')

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(project_output_file))


# In[54]:

challenge_output_file = "challenge_output.mp4"
challenge_video = VideoFileClip("challenge_video.mp4")

challenge_output = challenge_video.fl_image(pipeline_img)
get_ipython().magic(
    'time challenge_output.write_videofile(challenge_output_file, audio=False)')

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output_file))


# In[55]:


harder_challenge_output_file = "harder_challenge_output.mp4"
harder_challenge_video = VideoFileClip("harder_challenge_video.mp4")

harder_challenge_output = harder_challenge_video.fl_image(pipeline_img)
get_ipython().magic(
    'time harder_challenge_output.write_videofile(harder_challenge_output_file,\
	audio=False)')

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(harder_challenge_output_file))
