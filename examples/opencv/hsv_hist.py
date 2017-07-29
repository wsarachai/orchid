import cv2
import numpy as np
import matplotlib.pyplot as plt

image_file = '/Users/sarachaii/Desktop/jpg1000/DSC_0005x1000.jpg'

img = cv2.imread(image_file)
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hue, sat, val = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]

chans = (hue, sat, val)
colors = ("hue", "sat", "val")
nums = (1, 2, 3)
bins = (16, 4, 4)
ranges = ([0, 15], [0, 3], [0, 3])

plt.figure(figsize=(8,6))
plt.subplots_adjust(hspace=.5)
for (n, chan, color, bin, range) in zip(nums, chans, colors, bins, ranges):
    hist = cv2.calcHist([chan], [0], None, [bin], range)
    plt.subplot(310+n)
    plt.title(color)
    plt.hist(np.ndarray.flatten(hist), bins=bin)
    print np.ndarray.flatten(hist)

plt.show()