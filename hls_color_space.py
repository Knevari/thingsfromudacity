import cv2
import numpy as np
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("filename")

# args = parser.parse_args()

# image = cv2.imread(args.filename)
image = cv2.imread('images/solidWhiteCurve.jpg')

hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

h = hls[:, :, 0]
l = hls[:, :, 1]
s = hls[:, :, 2]

cv2.imshow("HLS", hls)
cv2.imshow("H", h)
cv2.imshow("L", l)
cv2.imshow("S", s)

cv2.waitKey(0)
cv2.destroyAllWindows()
