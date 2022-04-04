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

bboxes = []

im = cv.imread('images/img3.jpg')
template = np.zeros(im.shape[0:2])

lab_im = cv.cvtColor(im, cv.COLOR_BGR2LAB)

for i in range(im.shape[2]):
    im_channel = lab_im[:,:,i]
    # plt.imshow(im_channel)
    # plt.show()
    blurred = cv.blur(im_channel,(5,5))
    thresh = skimage.filters.threshold_otsu(blurred)
    segmented = blurred > thresh
    # segmented = segmented
    plt.imshow(segmented)
    cleared = skimage.segmentation.clear_border(segmented)
    label_image = skimage.measure.label(cleared, connectivity=2)
    for region in skimage.measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 300:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
            fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            template[minr:maxr, minc:maxc] += 1
    plt.show()
max_boxes = template == np.max(template)
plt.imshow(max_boxes)
plt.show()

print(im.shape)