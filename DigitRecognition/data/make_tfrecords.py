from __future__ import division, print_function
import os
from PIL import Image
import cPickle as pickle

import numpy as np
import tensorflow as tf

from DigitRecognition.data.bounding_boxes_to_target import bounding_boxes_to_target


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_tf_records(image_dir, bounding_box_file, destination, anchor_boxes, grid_rows=5):
    """
    :param image_dir: Directory containing all the image files
    :param bounding_box_file: Pickled object containing the bounding box definitions
    :param destination: Filename for resulting tfrecords file
    :return: Writes destination with tfrecords data
    """

    with open(bounding_box_file, 'rb') as f:
        bounding_box_dict = pickle.load(f)

    writer = tf.python_io.TFRecordWriter(destination)
    index = 0

    for image in os.listdir(image_dir):
        print(image)
        image_array = np.array(Image.open(image_dir + os.sep + image))

        # The reason to store image sizes is that we have to know sizes of images to later read raw serialized strings,
        # convert to 1d array and convert to shape that images need to have.
        height = image_array.shape[0]
        width = image_array.shape[1]

        bounding_boxes = bounding_box_dict[image]
        target = bounding_boxes_to_target(bounding_boxes, anchor_boxes, (width, height), grid_rows)

        image_raw = image_array.tostring()
        target_raw = target.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(image_raw),
            'target_raw': _bytes_feature(target_raw)}))

        writer.write(example.SerializeToString())

        index += 1
        print('.', end=' ')
        if index % 80 == 0:
            print('\n')


if __name__ == '__main__':
    image_dir = '/Users/joshuabrowning/Personal/Udacity/DigitRecognitionApp/input/train'
    bounding_box_file = '/Users/joshuabrowning/Personal/Udacity/DigitRecognitionApp/input/train_bounding_boxes.p'
    destination = '/Users/joshuabrowning/Personal/Udacity/DigitRecognitionApp/input/train.tfrecords'

    with open('/Users/joshuabrowning/Personal/Udacity/DigitRecognitionApp/input/3_anchor_boxes.p', 'rb') as f:
        anchor_boxes = pickle.load(f)

    make_tf_records(image_dir, bounding_box_file, destination, anchor_boxes)
