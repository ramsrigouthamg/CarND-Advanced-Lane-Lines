import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


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



if __name__ == "__main__":
    # mtx_matrix , dist_matrix = getCalibration()
    # testImage1 = cv2.imread("camera_cal/calibration1.jpg")
    # dst1 = cv2.undistort(testImage1, mtx_matrix, dist_matrix, None, mtx_matrix)
    # cv2.imwrite("output_images/undistorted_1.jpg", dst1)
    #
    # testImage2 = cv2.imread("test_images/straight_lines1.jpg")
    # dst2 = cv2.undistort(testImage2, mtx_matrix, dist_matrix, None, mtx_matrix)
    # cv2.imwrite("output_images/straight_lines1_output.jpg", dst2)

    testImage = cv2.imread("test_images/test6.jpg")
    # cv2.imshow('test',testImage)
    hls = cv2.cvtColor(testImage, cv2.COLOR_BGR2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    thresh_S = (100,255)
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
    cv2.imshow('image_Sobel', sxbinary)
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




