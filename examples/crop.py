import os
import shutil
from PIL import Image
from resizeimage import resizeimage
import glob

SRC_DIR = '/Users/sarachaii/Desktop/trains/resized/'
DST_DIR = '/Users/sarachaii/Desktop/trains/resized4/'

TRAIN_DIR = os.path.join(DST_DIR, 'train')
TEST_DIR = os.path.join(DST_DIR, 'test')

if os.path.isdir(TRAIN_DIR):
    shutil.rmtree(TRAIN_DIR)

if os.path.isdir(TEST_DIR):
    shutil.rmtree(TEST_DIR)

os.mkdir(TRAIN_DIR)
os.mkdir(TEST_DIR)

for dirid in range(12):

    #SAVE_DIR = DST_DIR + '/' + str(dirid+1)
    #SAVE_DIR = DST_DIR

    #if os.path.isdir(SAVE_DIR):
    #    shutil.rmtree(SAVE_DIR)

    #os.mkdir(SAVE_DIR)

    files = glob.glob(SRC_DIR + str(dirid+1) + "/*.jpg")

    fileid = 1

    for file in files:
        with open(file, 'r+b') as f:
            with Image.open(f) as img:
                w, h = img.size

                if w==224 and h==224:
                    pass
                else:
                    if w < h:
                        img = resizeimage.resize_width(img, 224)
                    else:
                        img = resizeimage.resize_height(img, 224)

                    img = resizeimage.resize_crop(img, [224, 224])

                if fileid > 80:
                    img.save(TEST_DIR + '/' +str(dirid) + '_' + str(fileid) + '.jpg', img.format)
                else:
                    img.save(TRAIN_DIR + '/' +str(dirid) + '_' + str(fileid) + '.jpg', img.format)
                fileid = fileid + 1
