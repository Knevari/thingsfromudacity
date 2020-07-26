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
combined = np.zeros_like(grad_binary)
combined[(grad_binary == 1) & (light_binary == 1)] = 1

y = image.shape[0]
x = image.shape[1]
offset = 100

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
Minv = cv2.getPerspectiveTransform(dst, src)

image = debugPolylines(image, src, dst)

warped = cv2.warpPerspective(
    combined, M, combined.shape[1::-1], flags=cv2.INTER_LINEAR)

histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)

plt.imshow(warped, cmap="gray")
plt.plot(histogram)
plt.show()
