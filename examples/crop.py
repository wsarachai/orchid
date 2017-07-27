import os
import shutil
from PIL import Image
from resizeimage import resizeimage
import glob

IMAGE_SIZE = 32
SRC_DIR = '/Users/sarachaii/Desktop/trains/raw_datas/processed/'
DST_DIR = '/Users/sarachaii/Desktop/trains/resized' + str(IMAGE_SIZE) + '/'

TRAIN_DIR = os.path.join(DST_DIR, 'train')
TEST_DIR = os.path.join(DST_DIR, 'test')

if os.path.isdir(TRAIN_DIR):
    shutil.rmtree(TRAIN_DIR)

if os.path.isdir(TEST_DIR):
    shutil.rmtree(TEST_DIR)

#os.mkdir(DST_DIR)
os.mkdir(TRAIN_DIR)
os.mkdir(TEST_DIR)

for dirid in range(12):

    #SAVE_DIR = DST_DIR + '/' + str(dirid+1)
    #SAVE_DIR = DST_DIR

    #if os.path.isdir(SAVE_DIR):
    #    shutil.rmtree(SAVE_DIR)

    #os.mkdir(SAVE_DIR)

    files = glob.glob(SRC_DIR + str(dirid+1) + "/*.png")

    fileid = 1

    for file in files:
        with open(file, 'r+b') as f:
            with Image.open(f) as img:
                w, h = img.size

                if w==IMAGE_SIZE and h==IMAGE_SIZE:
                    pass
                else:
                    if w < h:
                        img = resizeimage.resize_width(img, IMAGE_SIZE)
                    else:
                        img = resizeimage.resize_height(img, IMAGE_SIZE)

                    img = resizeimage.resize_crop(img, [IMAGE_SIZE, IMAGE_SIZE])

                if fileid > 80:
                    img.save(TEST_DIR + '/' +str(dirid) + '_' + str(fileid) + '.png', img.format)
                else:
                    img.save(TRAIN_DIR + '/' +str(dirid) + '_' + str(fileid) + '.png', img.format)
                fileid = fileid + 1
