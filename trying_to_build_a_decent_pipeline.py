import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def absoluteSobel(img, orient="x"):
    """Calculate the absolute sobel operator"""
    if orient == "x":
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    return np.uint8(255 * abs_sobel / np.max(abs_sobel))


def debugPolylines(img, src, dst):
    copy_img = np.copy(img)
    src_vertices = np.array([src], dtype=np.int32)
    dst_vertices = np.array([dst], dtype=np.int32)
    cv2.polylines(copy_img, src_vertices, True, (255, 0, 0), 4)
    cv2.polylines(copy_img, dst_vertices, True, (0, 0, 255), 4)
    return copy_img


def toHLS(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]


def getSobelBinary(img, sobel, sobel_thresh=(20, 150)):
    s_binary = np.zeros_like(img)
    s_binary[(sobel > sobel_thresh[0]) & (sobel <= sobel_thresh[1])] = 1
    return s_binary


def getLightnessBinary(img, light_min=210):
    light_binary = np.zeros_like(img)
    light_binary[L > light_min] = 1
    return light_binary


def getDirectionBinary(sobely, sobelx, dir_thresh=(np.pi/6, np.pi/2)):
    grad_dir = np.arctan2(sobely, sobelx)
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir > dir_thresh[0]) & (grad_dir <= dir_thresh[1])] = 1
    return dir_binary


def warpImage(img):
    y = img.shape[0]
    x = img.shape[1]
    offset = 40

    src = np.float32([
        [160,  720],
        [570,  460],
        [690,  460],
        [1090, 720]
    ])

    dst = np.float32([
        [offset, y-offset],
        [offset, offset],
        [x-offset, offset],
        [x-offset, y-offset]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    # Minv = cv2.getPerspectiveTransform(dst, src)

    img = debugPolylines(img, src, dst)

    warped = cv2.warpPerspective(
        combined, M, combined.shape[1::-1], flags=cv2.INTER_LINEAR)

    return warped


def divideHistogram(hist):
    midpoint = np.int(hist.shape[0]//2)
    leftx_base = np.argmax(hist[:midpoint])
    rightx_base = np.argmax(hist[midpoint:]) + midpoint
    return leftx_base, rightx_base


images = [
    "images/signs_vehicles_xygrad.png",
    "images/test.jpg",
    "images/solidYellowCurve2.jpg"
]
image = mpimg.imread(images[1])
image = cv2.resize(image, (1280, 720))

H, L, S = toHLS(image)

sobelx = absoluteSobel(S, "x")
sobely = absoluteSobel(S, "y")

light_binary = getLightnessBinary(L)
sx_binary = getSobelBinary(S, sobelx)
dir_binary = getDirectionBinary(sobely, sobelx)

grad_binary = np.zeros_like(H)
grad_binary[(sx_binary == 1) & (dir_binary == 1)] = 1

# combined = np.dstack(
#     (np.zeros_like(H), light_binary, grad_binary)) * 255

# Combine multiple thresholds to a single output
combined = np.zeros_like(grad_binary)
combined[(grad_binary == 1) & (light_binary == 1)] = 1

warped = warpImage(combined)

# Create a histogram to identify where the lane lines possibly are
histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)

# Apply sliding windows technique to find lanes
leftx_base, rightx_base = divideHistogram(histogram)

nonzero = warped.nonzero()
nonzeroy = nonzero[0]
nonzerox = nonzero[1]

# Number of sliding windows
nwindows = 9
margin = 100
minpix = 50

window_height = np.int(warped.shape[0]//nwindows)

leftx_current = leftx_base
rightx_current = rightx_base

out_image = np.dstack((warped, warped, warped)) * 255

left_lane_idxs = []
right_lane_idxs = []

for w in range(nwindows):
    y_top = warped.shape[0] - (w * window_height)
    y_bottom = warped.shape[0] - ((w + 1) * window_height)

    left_window_leftx = leftx_current - margin
    left_window_rightx = leftx_current + margin

    right_window_leftx = rightx_current - margin
    right_window_rightx = rightx_current + margin

    cv2.rectangle(out_image, (left_window_leftx, y_top),
                  (left_window_rightx, y_bottom), (0, 255, 0), 4)
    cv2.rectangle(out_image, (right_window_leftx, y_top),
                  (right_window_rightx, y_bottom), (0, 255, 0), 4)

    left_idxs = ((nonzeroy >= y_bottom) & (nonzeroy < y_top) &
                 (nonzerox >= left_window_leftx) & (nonzerox < left_window_rightx)).nonzero()[0]

    right_idxs = ((nonzeroy >= y_bottom) & (nonzeroy < y_top) &
                  (nonzerox >= right_window_leftx) & (nonzerox < right_window_rightx)).nonzero()[0]

    left_lane_idxs.append(left_idxs)
    right_lane_idxs.append(right_idxs)

    if len(left_idxs) > minpix:
        leftx_current = np.int(np.mean(nonzerox[left_idxs]))

    if len(right_idxs) > minpix:
        rightx_current = np.int(np.mean(nonzerox[right_idxs]))

plt.imshow(out_image)
plt.show()
