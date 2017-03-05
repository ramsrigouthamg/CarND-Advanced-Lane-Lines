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
            # img = cv2.drawChessboardCorners(img,(ncols,nrows),corners,ret)
            # cv2.imshow("temp",img)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # cv2.imshow("temp",dst)
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

def warp(img):
    img_size = (200,200)
    src = np.float32([[193,720],[586, 454],[701, 454],[1128,720]])
    dst = np.float32([[40, 200], [40, 0], [160, 0], [160, 200]])

    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img,M,img_size,flags=cv2.INTER_LINEAR)
    # cv2.line(warped, (40, 200), (40, 0), (255, 0, 0), 5)
    # cv2.line(warped, (260, 0), (260, 200), (255, 0, 0), 5)
    return warped,Minv



if __name__ == "__main__":
    mtx_matrix , dist_matrix = getCalibration()
    # testImage1 = cv2.imread("camera_cal/calibration1.jpg")
    # dst1 = cv2.undistort(testImage1, mtx_matrix, dist_matrix, None, mtx_matrix)
    # cv2.imwrite("output_images/undistorted_1.jpg", dst1)
    #
    # testImage2 = cv2.imread("test_images/straight_lines1.jpg")
    # dst2 = cv2.undistort(testImage2, mtx_matrix, dist_matrix, None, mtx_matrix)
    # cv2.imwrite("output_images/straight_lines1_output.jpg", dst2)

    testImage_in = cv2.imread("test_images/test6.jpg")
    testImage = cv2.undistort(testImage_in, mtx_matrix, dist_matrix, None, mtx_matrix)
    # cv2.imshow('test',testImage)
    hls = cv2.cvtColor(testImage, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    thresh_S = (170,250)
    binary_S = np.zeros_like(S)
    binary_S[(S > thresh_S[0]) & (S <= thresh_S[1])] = 255
    # cv2.imshow('image_S', binary_S)

    gray = cv2.cvtColor(testImage,cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    thresh_min = 30
    thresh_max = 150
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255

    combined = np.zeros_like(gray)
    combined[(sxbinary==255) | (binary_S==255)] = 1

    # cv2.imshow('image_combined', combined)
    # cv2.imshow('sobel',sxbinary)
    left_bottom = (193, 720)
    left_top = (586, 454)
    right_bottom = (1128, 720)
    right_top = (701, 454)

    # cv2.line(testImage, left_bottom,left_top , (255,0,0),5)
    # cv2.line(testImage, right_bottom,right_top , (255, 0, 0), 5)

    # cv2.imshow('Original', testImage)
    # cv2.imshow('S channel', binary_S)
    # cv2.imshow('Sobel', sxbinary)
    # cv2.imshow('Combined', combined)

    binary_warped,minV = warp(combined)
    plt.subplot(3,1,1)
    plt.imshow(binary_warped)

    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    out_img = np.dstack((binary_warped,binary_warped,binary_warped))*255

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
    plt.subplot(3, 1, 2)
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    # plt.plot(histogram)


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
    newwarp = cv2.warpPerspective(color_warp, minV, (testImage.shape[1], testImage.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(testImage, 1, newwarp, 0.3, 0)
    plt.subplot(3, 1, 3)
    plt.imshow(result)

    plt.show()
    # # plot_images(testImage, H, binary_S, S, 'original', 'H', 'binary_S', 'S')
    # cv2.namedWindow('image_S')
    # # cv2.namedWindow('image_H')
    # cv2.createTrackbar('S_lower', 'image_S', 0, 255, nothing)
    # cv2.createTrackbar('S_higher', 'image_S', 0, 255, nothing)
    # # cv2.createTrackbar('H_lower', 'image_H', 0, 255, nothing)
    # # cv2.createTrackbar('H_higher', 'image_H', 0, 255, nothing)
    # temp_S = np.zeros_like(S)
    # # temp_H = np.zeros_like(H)
    #
    # while (1):
    #
    #     cv2.imshow('image_S', temp_S)
    #     cv2.imshow('image_original', testImage)
    #     #cv2.imshow('image_H', temp_H)
    #     k = cv2.waitKey(1) & 0xFF
    #     if k == 27:
    #         break
    #     temp_S = np.zeros_like(S)
    #     # temp_H = np.zeros_like(H)
    #     s_low = cv2.getTrackbarPos('S_lower', 'image_S')
    #     s_high = cv2.getTrackbarPos('S_higher', 'image_S')
    #     h_low = cv2.getTrackbarPos('H_lower', 'image_H')
    #     h_high = cv2.getTrackbarPos('H_higher', 'image_H')
    #     temp_S[(S > s_low) & (S <= s_high)] = 255
        # temp_H[(H > h_low) & (H <= h_high)] = 255

    cv2.destroyAllWindows()




