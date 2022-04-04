import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.patches
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import skimage
import skimage.measure
import skimage.segmentation

bboxes = []

im1 = cv.imread('images/img4.jpg')

hsv_im = cv.cvtColor(im1, cv.COLOR_BGR2HSV)

blur_im = cv.blur(hsv_im,(5,5))
print(im1.shape)

light_red = (1, 100, 200)
dark_red = (18, 255, 255)

# lo_square = np.full((10, 10, 3), light_red, dtype=np.uint8) / 255.0
# do_square = np.full((10, 10, 3), dark_red, dtype=np.uint8) / 255.0

mask = cv.inRange(hsv_im, light_red, dark_red)

cleared = skimage.segmentation.clear_border(mask)

label_image = skimage.measure.label(cleared, connectivity=2)

res = im1[mask, :]

plt.imshow(mask)

for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 300:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
            fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)

plt.show()


