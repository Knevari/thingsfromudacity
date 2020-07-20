import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


def createVertices(img):
    ysize = img.shape[0]
    xsize = img.shape[1]
    vertices = np.array(
        [[(0, ysize), (460, 305), (490, 305), (xsize, ysize)]],
        dtype=np.int32
    )
    return vertices


def getEdgesFromImage(img):
    blurred_gray_image = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred_gray_image, 60, 190)

    mask = np.zeros_like(edges)
    vertices = createVertices(edges)
    cv2.fillPoly(mask, vertices, 255)

    masked_edges = cv2.bitwise_and(edges, mask)

    return masked_edges


def getLinesFromImage(img):
    # Hough Line Detection Parameters
    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_length = 20
    max_line_gap = 20
    lines = cv2.HoughLinesP(img, rho, theta, threshold,
                            np.array([]), min_line_length, max_line_gap)
    return lines


def main():
    parser = argparse.ArgumentParser(description="Do stuff")
    parser.add_argument("filename")

    args = parser.parse_args()

    image = mpimg.imread(args.filename)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = getEdgesFromImage(gray_image)
    lines = getLinesFromImage(edges)

    line_image = np.zeros_like(image)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    color_edges = np.dstack((edges, edges, edges))
    combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

    plt.imshow(combo)
    plt.show()


if __name__ == "__main__":
    main()
