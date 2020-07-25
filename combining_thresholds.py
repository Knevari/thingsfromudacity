import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


image = mpimg.imread('images/bridge_shadow.jpg')

# Edit this function to create your own pipeline.


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    image = np.copy(img)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    L = hls[:, :, 1]
    S = hls[:, :, 2]

    sobelx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_thresh[0]) &
              (scaled_sobel <= sx_thresh[1])] = 1

    s_binary = np.zeros_like(S)
    s_binary[(S >= s_thresh[0]) & (S <= s_thresh[1])] = 1

    combined = np.dstack((np.zeros_like(sx_binary), sx_binary, s_binary)) * 255

    return combined


result = pipeline(image, (170, 255), (20, 120))

# Plot the result
f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.set_title("Original Image")
ax1.imshow(image)

ax2.set_title("Pipeline Result")
ax2.imshow(result)

plt.show()
