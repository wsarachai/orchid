import os
import io
import re
import glob
import shutil
import tensorflow as tf
from tensorflow.python.platform import gfile

try:
    to_unicode = unicode
except NameError:
    to_unicode = str

def formatName(f, fid):
    return '{}_{:03d}{}'.format(f, fid, '.jpg')



def moveFile(src, dest):
    try:
        shutil.move(src, dest)
    # eg. src and dest are the same file
    except shutil.Error as e:
        print('Error: %s' % e)
    # eg. source or destination doesn't exist
    except IOError as e:
        print('Error: %s' % e.strerror)


def copyFile(src, dest):
    try:
        shutil.copy(src, dest)
    # eg. src and dest are the same file
    except shutil.Error as e:
        print('Error: %s' % e)
    # eg. source or destination doesn't exist
    except IOError as e:
        print('Error: %s' % e.strerror)


def modifyName(image_dir):
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None

  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue

    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))

    if not file_list:
      tf.logging.warning('No files found')
      continue

    #label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    #label_name = re.sub(r'[ ]', '-', label_name)

    label_name = dir_name.lower()

    fid = 1

    for f in file_list:
      new_name = formatName(label_name, fid)
      fid+=1

      new_name = os.path.join(sub_dir, new_name)
      moveFile(f, new_name)

if __name__ == "__main__":
  #modifyName('/Volumes/Data/_Corpus-data/orchid-3-type/flower_photos/')
  #modifyName('/Volumes/Data/_Corpus-data/orchids/pre-collect-data/dendrobium')
  modifyName('/Volumes/Data/_Corpus-data/orchids/all-orchid-dataset')