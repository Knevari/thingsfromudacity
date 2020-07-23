import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob


def main():
    nx = 8
    ny = 6

    img_points = []
    obj_points = []

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    calibration_imgs_folder = "images/calibration_wide/*"

    images = glob.glob(calibration_imgs_folder)

    for fname in images:
        image = cv2.imread(fname)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            img_points.append(corners)
            obj_points.append(objp)

    test_image = mpimg.imread("images/calibration_wide/test_image.jpg")

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, test_image.shape[1:], None, None)

    undistorted = cv2.undistort(test_image, mtx, dist, None, mtx)

    plt.imshow(undistorted)
    plt.show()


if __name__ == "__main__":
    main()
