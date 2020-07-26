import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import argparse


def getCombinedThresholds(image, color_thresh=(150, 255), sobel_thresh=(20, 100)):
    img = np.copy(image)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Apply the color threshold in the image
    color_binary = np.zeros_like(s_channel)
    color_binary[(s_channel >= color_thresh[0]) &
                 (s_channel < color_thresh[1])] = 1

    # Apply Sobel operator to get the image derivative
    sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= sobel_thresh[0]) &
                 (scaled_sobel < sobel_thresh[1])] = 1

    # return np.dstack((np.zeros_like(color_binary), color_binary, sobel_binary)) * 255

    combined = np.zeros_like(sobel_binary)
    combined[(sobel_binary == 1) | (color_binary == 1)] = 1
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Get bird view perspective of some image")
    parser.add_argument("filename")

    args = parser.parse_args()
    filename = args.filename

    # Read some image
    image = mpimg.imread(filename)
    combined = getCombinedThresholds(image)

    ysize = image.shape[0]
    xsize = image.shape[1]

    src = np.float32([[150, 538], [445, 325], [510, 325], [870, 538]])
    dst = np.float32([[0, ysize], [0, 0], [xsize, 0], [xsize, ysize]])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, image.shape[1::-1])

    f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.set_title("Original Image")
    ax1.imshow(image)

    ax2.set_title("Warped Image")
    ax2.imshow(warped)

    plt.show()


if __name__ == "__main__":
    main()
