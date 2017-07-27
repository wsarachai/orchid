import os
import shutil
from PIL import Image
import glob

SRC_DIR = '/Users/sarachaii/Desktop/trains/resized224/train/'
DST_DIR = '/Users/sarachaii/Desktop/trains/resized224-jpg/'

TRAIN_DIR = os.path.join(DST_DIR, 'train')
TEST_DIR = os.path.join(DST_DIR, 'test')

if os.path.isdir(DST_DIR):
    shutil.rmtree(DST_DIR)

os.mkdir(DST_DIR)

files = glob.glob(SRC_DIR + "/*.png")

for file in files:
    with open(file, 'r+b') as f:
        with Image.open(f) as img:
            h, l = os.path.split(file)
            l = l.split('.')[0] + '.jpg'
            save_file = os.path.join(DST_DIR, l)
            rgb_im = img.convert('RGB')
            rgb_im.save(save_file)
