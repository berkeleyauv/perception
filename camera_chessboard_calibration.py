import numpy as np
import cv2
import glob

##################################################################
# Measure characteristics of the camera so that
# later images can be undistorted with
# cv2.undistort() and camera matrix and dist values.
# Lines appear straighter. 
# Requires a certain number of image samples of a chessboard.
#  - add more images if the result is weird and swirly 
#
# It's possible to use a circle grid if chessboard function is changed:
# https://docs.opencv.org/3.4/d4/d94/tutorial_camera_calibration.html
#
# Most of the code is from:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
##################################################################


##################################################################
# Test 1: OpenCV's tutorial dataset
# source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
##################################################################
# images = glob.glob('../data/opencv_tutorial_calibration_images/*.jpg')
# test_img = cv2.imread('../data/opencv_tutorial_calibration_images/left02.jpg')
# checker_rows, checker_cols = 6, 7

##################################################################
# Test 2: Munich Visual Odometry dataset
# source: https://vision.in.tum.de/data/datasets/mono-dataset
##################################################################
images = glob.glob('../data/calib_narrow_checkerboard1/images/*.jpg')
test_img = cv2.imread('../data/sequence_47/images/00001.jpg')
checker_rows, checker_cols = 5, 8


def undistort_test(mtx, dist):
    h,  w = test_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(test_img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]

    cv2.imshow('original', test_img)
    cv2.imshow('undistorted',dst)
    print('camera matrix:')
    print(mtx)
    print('dist:')
    print(dist)
    cv2.waitKey(500)
    print('Hit enter to quit')
    input()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((checker_rows*checker_cols,3), np.float32)
objp[:,:2] = np.mgrid[0:checker_cols,0:checker_rows].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Gather data points
count = 0
for fname in images[:20]:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (checker_cols, checker_rows),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (checker_cols, checker_rows), corners2,ret)
        cv2.imshow('chessboard', img)

        cv2.waitKey(500)
    print(count)
    count += 1

# Get camera matrix and dist
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# Test it out
undistort_test(mtx, dist)

cv2.destroyAllWindows()