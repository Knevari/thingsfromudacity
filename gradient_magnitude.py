import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('images/signs_vehicles_xygrad.png')


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    mag = np.uint8(255 * mag / np.max(mag))

    sbinary = np.zeros_like(mag)
    sbinary[(mag >= mag_thresh[0]) & (mag <= mag_thresh[1])] = 1

    return sbinary


mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(mag_binary, cmap='gray')
ax2.set_title('Thresholded Magnitude', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
