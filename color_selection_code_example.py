import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read an image file
image = mpimg.imread('test.jpg')

# Print some image stats
print('This image is: ', type(image), image.shape)

# Grab the x and y sizes
ysize = image.shape[0]
xsize = image.shape[1]

# Make a copy of the image
color_select = np.copy(image)

# Define a color threshold in these variables
red_threshold = 210
green_threshold = 210
blue_threshold = 210

# Populate rgb_threshold
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Select any pixels below threshold and set them to 0
thresholds = (image[:, :, 0] < rgb_threshold[0]) \
    | (image[:, :, 1] < rgb_threshold[1]) \
    | (image[:, :, 2] < rgb_threshold[2])

print(type(thresholds))

color_select[thresholds] = [0, 0, 0]

plt.imshow(color_select)
plt.show()
