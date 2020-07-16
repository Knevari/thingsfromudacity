# lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ksize = 5

# First we need the image edges
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

blurred_gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
edges = cv2.Canny(blurred_gray, 50, 150)

# Hough Transform stuff
# rho and theta are the distance and angular resolution of the grid in H space
# rho needs to be in pixels and theta in radians
rho = 1
theta = (np.pi / 180)
threshold = 150
min_line_length = 100
max_line_gap = 12

# This creates a blank image with the same size of the og image
line_image = np.copy(image) * 0

# A loooot of parameters
# This function is supposed to return x1, y1, x2, y2 arrays
# representing lines
# Obs: Need to take care with parameters
lines = cv2.HoughLinesP(
    edges,
    rho,
    theta,
    threshold,
    np.array([]),
    min_line_length,
    max_line_gap
)

# Just go through the lines and draw 'em on the image
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

color_edges = np.dstack((edges, edges, edges))

combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

plt.imshow(combo)
plt.show()
