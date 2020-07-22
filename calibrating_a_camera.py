import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

images = glob.glob("images/calibration_wide/GOPR00*.jpg")

nx = 8
ny = 6

image_points = []
object_points = []

objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

for fname in images:
    image = mpimg.imread(fname)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ret, corners = cv2.findChessboardCorners(image, (nx, ny), None)

    if ret == True:
        image_points.append(corners)
        object_points.append(objp)

        # image = cv2.drawChessboardCorners(image, (nx, ny), corners, ret)

# Calibrate camera
test_image = mpimg.imread("images/calibration_wide/test_image.jpg")

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, test_image.shape[1:], None, None)

undistorted = cv2.undistort(test_image, mtx, dist, None, mtx)

plt.imshow(undistorted)
plt.show()
