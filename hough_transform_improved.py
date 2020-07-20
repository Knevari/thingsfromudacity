import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ksize = 5

# Read in and grayscale the image
image = mpimg.imread('images/exit-ramp.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Apply Gaussian Blur on the gray image and apply Canny
blur_gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)
edges = cv2.Canny(blur_gray, 50, 150)

# Create masked edges
mask = np.zeros_like(edges)
ignore_mask_color = 255

# Create a four sided polygon to mask
ysize = image.shape[0]
xsize = image.shape[1]

vertices = np.array(
    [[(0, ysize), (450, 290), (490, 290), (xsize, ysize)]],
    dtype=np.int32
)

cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# Hough transform parameters
rho = 2
theta = np.pi / 180
threshold = 15
min_line_length = 40
max_line_gap = 20

line_image = np.copy(image) * 0

lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

color_edges = np.dstack((edges, edges, edges))
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

plt.imshow(lines_edges)
plt.show()
