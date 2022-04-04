import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.patches
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import cv2 as cv

# takes a color image
# returns a list of bounding boxes and black_and_white image
bw = None
bboxes = []
# insert processing in here
# one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
# this can be 10 to 15 lines of code using skimage functions
# apply threshold

im1 = cv.imread('images/img4.jpg')

hsv_im = cv.cvtColor(im1, cv.COLOR_BGR2HSV)

# print(image.shape)

# denoised = skimage.restoration.denoise_tv_chambolle(image, weight=0.1, multichannel=True)
# print("complete")

# Blurring image (denoise takes too long)
blurred = skimage.filters.gaussian(im1, sigma=(1.5, 1.5), multichannel=True)


# Transferring to greyscale
greyscale = skimage.color.rgb2gray(blurred)


# Thresholding image based on otsu
thresh = skimage.filters.threshold_otsu(greyscale)

# Connecting open elements
eroded = skimage.morphology.erosion((greyscale > thresh), skimage.morphology.square(4))

bw = skimage.morphology.opening(eroded, skimage.morphology.square(6))

# Invert
bw = ~bw

# remove artifacts connected to image border
cleared = skimage.segmentation.clear_border(bw)

# label image regions
label_image = skimage.measure.label(cleared, connectivity=2)
# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`

for region in skimage.measure.regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 300:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        bboxes.append((minr, minc, maxr, maxc))

for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)

plt.imshow(greyscale)
plt.show()

print("complete")