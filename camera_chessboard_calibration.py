import numpy as np
import cv2
import glob, os
import random, string

##################################################################
# Measures characteristics of the camera so that
# later images can be undistorted with
# cv2.undistort(), camera matrix, and distortion values.
# Undistorted lines appear straighter and aid in feature detection. 
# Requires a certain number of image samples of a chessboard in a variety
# of positions and angles.
#  - add more diverse images if undistort_test errors or the result is swirly
#
# It's possible to use a circle grid instead of a chessboard:
# https://docs.opencv.org/3.4/d4/d94/tutorial_camera_calibration.html
#
# Most of the code is from:
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
##################################################################

##################################################################
# Helper functions
##################################################################

def isFromWebcam(cap):
    return isinstance(cap, cv2.VideoCapture)
def isFromFileSystem(cap):
    return isinstance(cap, list)
def get_frame(cap, index=0):
    """ Returns ret, frame just like a regular cv2.VideoCapture does"""
    if isFromFileSystem(cap):
        if len(cap) == 0 or index >= len(cap):
            return (False, None)
        else:
            return (True, cv2.imread(cap[index]))
    elif isFromWebcam(cap):
        return cap.read()
    else:
        return (True, cap)

##################################################################
# Test 1: OpenCV's tutorial dataset
# source: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
##################################################################
# images = glob.glob('../data/opencv_tutorial_calibration_images/*.jpg')
# test_img = get_frame(cv2.imread('../data/opencv_tutorial_calibration_images/left02.jpg'))
# checker_rows, checker_cols = 6, 7

##################################################################
# Test 2: Munich Visual Odometry dataset
# source: https://vision.in.tum.de/data/datasets/mono-dataset
##################################################################
# images = glob.glob('../data/calib_narrow_checkerboard1/images/*.jpg')
# test_img = get_frame(cv2.imread('../data/sequence_47/images/00001.jpg'))
# checker_rows, checker_cols = 5, 8

##################################################################
# Test 3: Pictures taken by oneself
##################################################################
# # Files from my computer 1
# images = glob.glob('iphone_chessboard_imgs/*.JPG')
# test_img = get_frame(cv2.imread('iphone_chessboard_imgs/IMG_3413.JPG'))
# checker_rows, checker_cols = 7, 7
# # Files from my computer 2
# images = glob.glob('*.png')
# test_img = get_frame(cv2.imread('GIKTZTK9HV.png'))
# checker_rows, checker_cols = 7, 7
# Use webcam
images = cv2.VideoCapture(0)
test_img = get_frame(cv2.VideoCapture(0))
checker_rows, checker_cols = 7, 7

def undistort_test(mtx, dist, test_img):
    print('camera matrix:')
    print(np.array2string(mtx, separator=', '))
    print('distortion matrix:')
    print(np.array2string(dist, separator=', '))

    ret, test_img = get_frame(test_img)
    h,  w = test_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(test_img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]

    scaling = 500/max(test_img.shape)
    cv2.imshow('original', cv2.resize(test_img, None, fx=scaling, fy=scaling))
    if dst.shape[:2] != (0, 0):
        cv2.imshow('undistorted', cv2.resize(dst, None, fx=scaling, fy=scaling))
    else:
        print('Error: No valid undistort_test result')
    cv2.waitKey(500)
    print('Hit enter to quit')
    input()

def is_recording():
    print("Do you want to save the images taken in this session? (y/n)")
    user = input()
    if 'y' in user or 't' in user:
        return True
    if 'n' in user or 'f' in user:
        return False
    else:
        return is_recording()

########################################################################
# Start script
########################################################################

if isFromWebcam(images):
    print("Is recording")
    recording = is_recording()
else:
    print("Not recording")
    recording = False

# subpix function termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((checker_rows*checker_cols,3), np.float32)
objp[:,:2] = np.mgrid[0:checker_cols,0:checker_rows].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Gather data points
count = 0
ret_frame = True
gray = None
while count < 40 and ret_frame:
    ret_frame, img = get_frame(images, count)
    if ret_frame:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (checker_cols, checker_rows),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img_chess = np.copy(img)
            cv2.drawChessboardCorners(img_chess, (checker_cols, checker_rows), corners2,ret)

            cv2.imshow('chessboard', img_chess)
            if recording:
                name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
                cv2.imwrite(name + '.png', img)
            print(count)
            cv2.waitKey(500)
        else:
            cv2.imshow('chessboard', img)
            print(count, 'no chessboard found. skipped', images[count].split('/').pop() if isinstance(images, list) else '') 
            if isinstance(images, list):
                os.remove('iphone_chessboard_imgs/'+images[count].split('/').pop())
                print(' -removed')
            cv2.waitKey(500)
    count += 1

if gray is not None:
    # Get camera matrix and dist
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    # Test it out
    undistort_test(mtx, dist, test_img)
else:
    print("No images found. If you want to capture from your webcam, uncomment the 'Use webcam' block at the top")

cv2.destroyAllWindows()
