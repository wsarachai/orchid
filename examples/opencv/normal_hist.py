import cv2
import matplotlib.pyplot as plt

image_file = '/Users/sarachaii/Desktop/jpg1000/DSC_0005x1000.jpg'

img = cv2.imread(image_file)

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
