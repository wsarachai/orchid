import cv2

import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

image_file = '/Users/sarachaii/Desktop/jpg1000/DSC_0005x1000.jpg'

img = cv2.imread(image_file)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 15, 0, 3, 0, 3])
#hist = cv2.calcHist([img],[0],None,[256],[0,256])

#plt.hist(hist.ravel(),256,[0,256]);
#plt.show()

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

#SFTA = sfta.SegmentationFractalTextureAnalysis(8)
#fv = SFTA.feature_vector(gray)

#sift = cv2.xfeatures2d.SIFT_create()

#print len(fv)

#kp = sift.detect(gray, None)

#img = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#cv2.imwrite('/Users/sarachaii/Desktop/jpg1000/sift_keypoints.jpg',img)