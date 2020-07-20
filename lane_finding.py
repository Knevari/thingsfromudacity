import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


def gaussian_blur(image, ksize=3, σ=0):
    return cv2.GaussianBlur(image, (ksize, ksize), σ)


def draw_line(image, line, color=[255, 0, 0], thickness=10):
    x1, y1, x2, y2 = line
    cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def divide_lines(lines, dimensions=np.array([[3, 3, 3]], dtype=np.uint8)):
    left_lines = []
    right_lines = []

    left_lines_len = []
    right_lines_len = []

    left_avg_slope = 0
    left_avg_intercept = 0

    right_avg_slope = 0
    right_avg_intercept = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x1 == x2:
            continue
        if y1 == y2:
            continue

        # Unicode just bcz I can
        Δx = float(x2 - x1)
        Δy = float(y2 - y1)

        m = Δy / Δx

        # b = y - mx
        b = y1 - m * x1

        # Euclidean Distance
        d = np.sqrt(np.power(Δx, 2) + np.power(Δy, 2))

        if m < 0:
            left_lines_len.append(d)
            left_lines.append((m, b))
        else:
            right_lines_len.append(d)
            right_lines.append((m, b))

    if left_lines:
        left_lines_dot = np.dot(left_lines_len, left_lines)
        left_lines_len_sum = np.sum(left_lines_len)
        left_avg_slope, left_avg_intercept = left_lines_dot / left_lines_len_sum

    if right_lines:
        right_lines_dot = np.dot(right_lines_len, right_lines)
        right_lines_len_sum = np.sum(right_lines_len)
        right_avg_slope, right_avg_intercept = right_lines_dot / right_lines_len_sum

    y_starting_point = int(dimensions[0])
    y_ending_point = int(dimensions[0] * 0.6)

    left_x1 = int((y_starting_point - left_avg_intercept) / left_avg_slope)
    left_x2 = int((y_ending_point - left_avg_intercept) / left_avg_slope)

    right_x1 = int((y_starting_point - right_avg_intercept) / right_avg_slope)
    right_x2 = int((y_ending_point - right_avg_intercept) / right_avg_slope)

    left_lane = (left_x1, y_starting_point, left_x2, y_ending_point)
    right_lane = (right_x1, y_starting_point, right_x2, y_ending_point)

    return left_lane, right_lane


def draw_lines(image, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def hough_lines(image, ρ, ϴ, threshold, min_line_len, max_line_gap):
    """Image needs to be canny edges detection output"""
    lines = cv2.HoughLinesP(image, ρ, ϴ, threshold,
                            np.array([]), min_line_len, max_line_gap)
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    left_lane, right_lane = divide_lines(lines, image.shape)

    # BGR
    draw_line(line_img, left_lane, [0, 0, 255])
    draw_line(line_img, right_lane, [255, 0, 0])
    # draw_lines(line_img, left_lines, [0, 255, 0])
    # draw_lines(line_img, right_lines, [0, 0, 255])

    return line_img


def get_roi(image):
    mask = np.zeros_like(image)

    ysize = image.shape[0]
    xsize = image.shape[1]

    left_bottom = (0, ysize)
    left_top = (xsize / 2 - 60, ysize / 2 + 60)

    right_bottom = (xsize, ysize)
    right_top = (xsize / 2 + 60, ysize / 2 + 60)

    vertices = np.array(
        [[left_bottom, left_top, right_top, right_bottom]],
        dtype=np.int32
    )

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(mask, image)
    return masked_image


def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = gaussian_blur(gray, 9)
    edges = cv2.Canny(blur, 50, 150)
    roi = get_roi(edges)
    lines = hough_lines(roi, 2, np.pi / 180, 15, 5, 20)
    output = cv2.addWeighted(image, .8, lines, .8, 0.)
    return output


def main():
    parser = argparse.ArgumentParser(description="Do Stuff")
    parser.add_argument("filename")

    args = parser.parse_args()

    filename, extension = args.filename.split('.')

    cap = cv2.VideoCapture(filename + '.' + extension)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('output/' + filename + '.mp4', fourcc, fps,
                          (frame_width, frame_height), True)

    while cap.isOpened():
        _, frame = cap.read()

        output = process_image(frame)
        out.write(output)

        cv2.imshow("Original Frame", frame)
        cv2.imshow("Modified Frame", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
