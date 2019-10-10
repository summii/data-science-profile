import os
import os.path
import warnings

from skimage import image_as_ubyte
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf

images_directory = "/Users/sumeet/Downloads/mirflickr"
tfrecords_subdir = "tfrecords"


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    
def process_image_dir(img_dir):
    
    count = 0
    
    tfrecords_filename = "mirflicker1m_images_{}.tfrecord"
    
    tfrecrds_path = os.path.join(images_directory, tfrecords_subdir, tfrecords_subdir.format(count))
    
    count = count+1
    tfrecords_writer = tf.python_io.TFRecordWriter(tfrecords_path)
    
    print("tfrecords file:", tfrecords_path)