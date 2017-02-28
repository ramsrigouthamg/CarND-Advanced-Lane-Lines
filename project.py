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










if __name__ == "__main__":
    mtx_matrix , dist_matrix = getCalibration()
    testImage = cv2.imread("camera_cal/calibration1.jpg")
    dst = cv2.undistort(testImage, mtx_matrix, dist_matrix, None, mtx_matrix)
    cv2.imwrite("output_images/undistorted_1.jpg", dst)