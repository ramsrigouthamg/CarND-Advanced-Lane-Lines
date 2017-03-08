import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from PIL import Image


def getCalibration():
    images = glob.glob('camera_cal/calibration*.jpg')
    objpoints = []
    imgpoints=[]
    nrows= 6
    ncols = 9
    objp = np.zeros((nrows*ncols,3),np.float32)
    objp[:,:2] = np.mgrid[0:ncols,0:nrows].T.reshape(-1,2)
    counter = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,corners = cv2.findChessboardCorners(gray,(ncols,nrows),None)
        if ret:
            counter = counter + 1
            imgpoints.append(corners)
            objpoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx,dist


def plot_images(original,img1,img2,img3,orig_label,label1,label2,label3):
    b, g, r = cv2.split(original)  # get b,g,r
    rgb_img = cv2.merge([r, g, b])
    plt.subplot(2, 2, 1)
    plt.imshow(rgb_img)
    plt.xlabel(orig_label)

    plt.subplot(2, 2, 2)
    plt.imshow(img1,cmap='gray')
    plt.xlabel(label1)
    plt.subplot(2, 2, 3)
    plt.imshow(img2,cmap='gray')
    plt.xlabel(label2)
    plt.subplot(2, 2, 4)
    plt.imshow(img3,cmap='gray')
    plt.xlabel(label3)
    plt.show()


def nothing(x):
    pass

def getPerspectiveTransformParameters():
    src = np.float32([[193, 720], [586, 454], [701, 454], [1128, 720]])
    dst = np.float32([[40, 200], [40, 0], [160, 0], [160, 200]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M,Minv


def warp(img,M):
    img_size = (200,200)
    warped = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
    return warped


def undistort_image(img,mtx_matrix,dist_matrix):
    undistort = cv2.undistort(img, mtx_matrix, dist_matrix, None, mtx_matrix)
    return undistort


def processImage(img):
    # undistort = cv2.undistort(img, mtx_matrix, dist_matrix, None, mtx_matrix)

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    thresh_S = (170, 250)
    binary_S = np.zeros_like(S)
    binary_S[(S > thresh_S[0]) & (S <= thresh_S[1])] = 255

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    sobelx_SChannel = cv2.Sobel(S, cv2.CV_64F, 1, 0)
    abs_sobelx_SChannel = np.absolute(sobelx_SChannel)
    scaled_sobel_SChannel = np.uint8(255 * abs_sobelx_SChannel / np.max(abs_sobelx_SChannel))

    thresh_min = 30
    thresh_max = 150
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255
    sxbinary_Schannel = np.zeros_like(scaled_sobel)
    sxbinary_Schannel[(scaled_sobel_SChannel >= thresh_min) & (scaled_sobel_SChannel <= thresh_max)] = 255
    combined = np.zeros_like(gray)
    combined[(sxbinary == 255) | (binary_S == 255) | (sxbinary_Schannel == 255)] = 1

    return combined

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #Left fit
        self.left_fit = None
        #Right fit
        self.right_fit = None

def calculateCurvature(warped_image,original,MinV,lineObject):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
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
    margin = 25
    # Set minimum number of pixels found to recenter window
    minpix = 10
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    if lineObject.detected:
        margin = 25
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

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.subplot(3, 1, 2)
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')




    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, MinV, (original.shape[1], original.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(original, 1, newwarp, 0.3, 0)

    # plt.subplot(3, 1, 3)
    # plt.imshow(result)
    #
    # plt.show()

    # Radius of Curvature
    y_eval = np.max(ploty)
    # left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    # right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    # print(left_curverad, right_curverad)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    if abs(left_curverad-right_curverad) <200:
        lineObject.detected = True
        lineObject.left_fit = left_fit
        lineObject.right_fit = right_fit
    else:
        lineObject.detected = False

    return result,out_img,left_curverad,right_curverad



if __name__ == "__main__":
    mtx_matrix , dist_matrix = getCalibration()
    M,MinV = getPerspectiveTransformParameters()
    plotImages = False

    line = Line()
    # original = cv2.imread("test_images/test2.jpg")
    # original_undistorted =undistort_image(original,mtx_matrix,dist_matrix)
    # combined = processImage(original_undistorted)
    # binary_warped = warp(combined,M)
    # final_output,laneLines,left_curvature,right_curvature = calculateCurvature(binary_warped,original_undistorted,MinV)

    # if plotImages:
    #     plt.subplot(3, 2, 1)
    #     plt.imshow(cv2.cvtColor(original_undistorted, cv2.COLOR_BGR2RGB))
    #     plt.subplot(3, 2, 2)
    #     plt.imshow(combined)
    #     plt.subplot(3, 2, 3)
    #     plt.imshow(binary_warped)
    #     plt.subplot(3, 2, 4)
    #     plt.imshow(cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB))
    #     plt.subplot(3, 2, 5)
    #     plt.imshow(laneLines)
    #     plt.show()
    # print ('left curvature: ',left_curvature,'right curvature: ',right_curvature)

    cap = cv2.VideoCapture('project_video.mp4')

    while (cap.isOpened()):
        ret, frame = cap.read()
        print("line detected", line.detected)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        original_undistorted = undistort_image(frame, mtx_matrix, dist_matrix)
        combined = processImage(original_undistorted)
        binary_warped = warp(combined, M)
        final_output, laneLines, left_curvature, right_curvature = calculateCurvature(binary_warped,
                                                                                      original_undistorted, MinV,line)

        print ('left_curvature ',left_curvature,' right_curvature',right_curvature)

        font = cv2.FONT_HERSHEY_SIMPLEX
        textString = ' left_curvature: ' + str(round(left_curvature,2)) + ' mts' + ' right_curvature: '+ str(round(right_curvature,2)) +' mts'
        cv2.putText(final_output, textString, (100, 100), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', final_output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


