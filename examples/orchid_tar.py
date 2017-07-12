import os
import tarfile
import tensorflow as tf
import glob

EXT = '*.pickle'
DATA_DIR = '/Users/sarachaii/Desktop/trains/orchid11_data/orchid-11-batches-bin/'
TAR_DIR = '/Users/sarachaii/Desktop/trains/orchid11_data/'

def main(argv=None):
    tar = tarfile.open(os.path.join(TAR_DIR, "orchid-11-binary.tar.gz"), "w:gz")
    files = glob.glob(os.path.join(DATA_DIR, EXT))
    for name in files:
        tar.add(name)
        print (name)
    tar.close()

if __name__ == '__main__':
    tf.app.run()