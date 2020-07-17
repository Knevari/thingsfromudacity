import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('test1.jpg')

ysize = image.shape[0]
xsize = image.shape[1]

color_select = np.copy(image)
line_image = np.copy(image)

rgb_threshold = [200] * 3

left_bottom = [130, 540]
right_bottom = [850, 540]
apex = [450, 300]

fit_left = np.polyfit(
    (left_bottom[0], apex[0]),
    (left_bottom[1], apex[1]),
    1
)

fit_right = np.polyfit(
    (right_bottom[0], apex[0]),
    (right_bottom[1], apex[1]),
    1
)

fit_bottom = np.polyfit(
    (left_bottom[0], right_bottom[0]),
    (left_bottom[1], right_bottom[1]),
    1
)

XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))

color_thresholds = (image[:, :, 0] < rgb_threshold[0]) | \
    (image[:, :, 1] < rgb_threshold[1]) | \
    (image[:, :, 2] < rgb_threshold[2])

region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                    (YY > (XX * fit_right[0] + fit_right[1])) & \
                    (YY < (XX * fit_bottom[0] + fit_bottom[1]))

color_select[color_thresholds | ~region_thresholds] = [0, 0, 0]
line_image[~color_thresholds & region_thresholds] = [255, 0, 0]

plt.imshow(image)
x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]
y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]
plt.plot(x, y, 'b--', lw=4)
plt.imshow(color_select)
plt.imshow(line_image)
plt.show()
