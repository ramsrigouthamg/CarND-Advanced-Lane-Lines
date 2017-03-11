import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from PIL import Image
from collections import deque

def getCalibration():
    """   Returns calibration parameters by processing the images used for calibration     """
    # Read all the calibration images from the folder
    images = glob.glob('camera_cal/calibration*.jpg')
    #Arrays to store object points and image points
    objpoints = [] #3D points in real world space
    imgpoints=[]  # 2D points in image plane
    nrows= 6  # No of inside corner rows in chess board
    ncols = 9 # No of inside corner columns in chessboard.
    # Prepare object points like (0,0,0),(1,0,0) ......
    objp = np.zeros((nrows*ncols,3),np.float32)
    objp[:,:2] = np.mgrid[0:ncols,0:nrows].T.reshape(-1,2) #x and y coordinates
    counter = 0
    for fname in images:
        img = cv2.imread(fname)
        # convert image to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret,corners = cv2.findChessboardCorners(gray,(ncols,nrows),None)
        # If corners are found add object points and image points
        if ret:
            counter = counter + 1
            imgpoints.append(corners)
            objpoints.append(objp)
    # Calculate distortion coefficients(dist) and camera matrix (mtx) using calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx,dist

def getPerspectiveTransformParameters():
    # Four source point defining the four corners of lane lines when car is driving straight
    src = np.float32([[193, 720], [586, 454], [701, 454], [1128, 720]])
    # Four destination points that form a rectangle in the perspective transformed image.
    dst = np.float32([[250, 720], [250, 0], [1030, 0], [1030, 720]])
    # Compute the perspective  transform, M
    M = cv2.getPerspectiveTransform(src, dst)
    # Compute the inverse perspective transform, MinV
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv

def warp(img,M):
    img_size = (1280,720)
    # Create a warped image - uses linear interpolation
    warped = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
    return warped

def undistort_image(img,mtx_matrix,dist_matrix):
    # undistort a given image taking distortion coefficients and camera matrix from calibration
    undistort = cv2.undistort(img, mtx_matrix, dist_matrix, None, mtx_matrix)
    return undistort

def processImage(img):
    # Convert input image to HLS colorspace
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # Separate H,L,S channels.
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    # Threshold to use for S channel image
    thresh_S = (170, 250)
    binary_S = np.zeros_like(S)
    # Get a binary image applying threshold to S channel image.
    binary_S[(S > thresh_S[0]) & (S <= thresh_S[1])] = 255

    #  Get a grayscale image from the input image.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x gradient with sobel function on gray image and take absolute value
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # Apply x gradient with sobel function on S channel and take absolute value
    sobelx_SChannel = cv2.Sobel(S, cv2.CV_64F, 1, 0)
    abs_sobelx_SChannel = np.absolute(sobelx_SChannel)
    # Rescale back to 8 bit integer
    scaled_sobel_SChannel = np.uint8(255 * abs_sobelx_SChannel / np.max(abs_sobelx_SChannel))
    # Threshold to use for Sobel gray and Sobel S channel images.
    thresh_min = 30
    thresh_max = 150
    # Get a binary image by applying threshold to Sobel x gradient gray image
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255
    # Get a binary image by applying threshold to Sobel x gradient S channel image
    sxbinary_Schannel = np.zeros_like(scaled_sobel)
    sxbinary_Schannel[(scaled_sobel_SChannel >= thresh_min) & (scaled_sobel_SChannel <= thresh_max)] = 255
    # Combine thresholded images of S channel, Sobel gray , Sobel S channel to yield a single image.
    combined = np.zeros_like(gray)
    combined[(sxbinary == 255) | (binary_S == 255) | (sxbinary_Schannel == 255)] = 1
    return combined

# Define a class to store the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        #Smoothing buffer to store the last 10 left fit pixels.
        self.left_fitx_buffer = deque(maxlen=10)
        # Store last left fit.
        self.left_fit = None
        # Smoothing buffer to store the last 10 right fit pixels.
        self.right_fitx_buffer = deque(maxlen=10)
        # Store last right fit.
        self.right_fit = None

def calculateCurvature(warped_image,original,MinV,lineObject):
    # Take histogram along the columns in the lower half of the image.
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
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
    margin = 75
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    #  If line was detected in the previous frame skip sliding windows and use a search margin.
    if lineObject.detected:
        margin = 50
        # Identify the nonzero pixels in x and y within the window
        left_lane_inds = ((nonzerox > (lineObject.left_fit[0] * (nonzeroy ** 2) + lineObject.left_fit[1] * nonzeroy + lineObject.left_fit[2] - margin)) & (
        nonzerox < (lineObject.left_fit[0] * (nonzeroy ** 2) + lineObject.left_fit[1] * nonzeroy + lineObject.left_fit[2] + margin)))
        right_lane_inds = (
        (nonzerox > (lineObject.right_fit[0] * (nonzeroy ** 2) + lineObject.right_fit[1] * nonzeroy + lineObject.right_fit[2] - margin)) & (
        nonzerox < (lineObject.right_fit[0] * (nonzeroy ** 2) + lineObject.right_fit[1] * nonzeroy + lineObject.right_fit[2] + margin)))
    else:
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
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
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
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    #  Append the left and right lane fits to the smoothing buffer in line class
    lineObject.left_fitx_buffer.appendleft(left_fitx)
    lineObject.right_fitx_buffer.appendleft(right_fitx)

    left_fitx = np.zeros_like(right_fitx)
    right_fitx = np.zeros_like(left_fitx)

    # Get new left_fitx and right_fitx which is the average of the last 10 detections
    for left_fitx_elem in lineObject.left_fitx_buffer:
        left_fitx = np.add(left_fitx ,left_fitx_elem)

    for right_fitx_elem in lineObject.right_fitx_buffer:
        right_fitx = np.add(right_fitx , right_fitx_elem)

    left_fitx = np.divide(left_fitx,len(lineObject.left_fitx_buffer))
    right_fitx = np.divide(right_fitx,len(lineObject.right_fitx_buffer))

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, MinV, (original.shape[1], original.shape[0]))

    #  Calculate center offset
    lastRow = newwarp[719,:,1]
    index = np.where(lastRow!=0)
    x_min = np.amin(index)
    x_max = np.amax(index)
    x_mid = (x_min + x_max)//2
    x_mid_image = 640
    dashCamCenterOffset = (x_mid - x_mid_image) * xm_per_pix

    # Combine the result with the original image
    result = cv2.addWeighted(original, 1, newwarp, 0.3, 0)

    # Radius of Curvature
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # If the difference of radius of curvature is within certain value consider that line was detected and append to line object.
    if abs(left_curverad-right_curverad) <400:
        lineObject.detected = True
        lineObject.left_fit = left_fit
        lineObject.right_fit = right_fit

    else:
        lineObject.detected = False

    return result,out_img,left_curverad,right_curverad,dashCamCenterOffset



if __name__ == "__main__":
    # Run calibration and get distortion coefficient and camera matrix.
    mtx_matrix , dist_matrix = getCalibration()
    #  Get perspective transform and inverse perspective transform parameters.
    M,MinV = getPerspectiveTransformParameters()
    # Create line object to store previous detections.
    line = Line()
    # Videocapture object to read the video.
    cap = cv2.VideoCapture('project_video.mp4')#.subclip(20,25)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))

    while (cap.isOpened()):
        # Read an image
        ret, frame = cap.read()
        # Undistort the image
        original_undistorted = undistort_image(frame, mtx_matrix, dist_matrix)
        # Get a binary image applying S channel and sobel thresholds.
        combined = processImage(original_undistorted)
        # Warp the image.
        binary_warped = warp(combined, M)
        # Get output image, curvature and offset values
        final_output, laneLines, left_curvature, right_curvature,offset = calculateCurvature(binary_warped,
                                                                                      original_undistorted, MinV,line)

        # Write curvature and offset values onto the image.
        font = cv2.FONT_HERSHEY_SIMPLEX
        textString = 'left_curvature: ' + str(round(left_curvature,2)) + ' m' + ' right_curvature: '+ str(round(right_curvature,2)) +' m'
        cv2.putText(final_output, textString, (100, 100), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        offsetString = 'Center Offset: '+str(round(offset,3))+ ' m'
        cv2.putText(final_output, offsetString, (100, 150), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('frame', final_output)
        # Write the final image to videoWriter
        out.write(final_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


