from __future__ import division

import numpy as np

NUM_CLASSES = 10


def bounding_boxes_to_target(bounding_boxes, anchor_boxes, image_size, grid_rows=3):
    """
    Per https://medium.com/@vivek.yadav/part-1-generating-anchor-boxes-for-yolo-like-network-for-vehicle-detection-using-kitti-dataset-b2fe033e5807:
    Note that the anchor box that is responsible to predict a ground truth label is chosen as the box that gives
    maximum IOU when placed at the center of the ground truth box, i.e. only size is considered while assigning ground
    truth boxes.
    :param bounding_boxes: List of bounding boxes [(target number, x position, y position, width, height)
    :param anchor_boxes: List of anchor box dimensions
    :param image_size: Size of image as (width, height)
    :param grid_rows: Number of rows; grid will be grid_rows x grid_rows
    :return: The target tensor
    """
    # 1 for yes/no object, 4 for bounding box shape, NUM_CLASSES for which class
    length_of_full_target = (1 + 4 + NUM_CLASSES)
    target = np.zeros((grid_rows, grid_rows, len(anchor_boxes) * length_of_full_target))
    anchor_boxes_used = np.zeros((grid_rows, grid_rows, len(anchor_boxes)))

    for bounding_box in bounding_boxes:
        label, x, y, width, height = bounding_box
        x_grid_coord = get_index_in_range(x + width/2, image_size[0], grid_rows)
        y_grid_coord = get_index_in_range(y + height/2, image_size[1], grid_rows)

        available_anchor_boxes = [(index, a_box) for index, a_box, avail in
                                  zip(range(len(anchor_boxes)), anchor_boxes,
                                            anchor_boxes_used[x_grid_coord, y_grid_coord, :])
                                  if avail == 0]
        anchor_box_sub_index = get_anchor_box_index([a_box for index, a_box in available_anchor_boxes], (width, height))
        anchor_box_index = available_anchor_boxes[anchor_box_sub_index][0]
        anchor_boxes_used[x_grid_coord, y_grid_coord, anchor_box_index] = 1

        start = anchor_box_index * length_of_full_target
        end = (anchor_box_index + 1) * length_of_full_target
        label_dummy = [0] * NUM_CLASSES
        label_dummy[label % 10] = 1  # For some reason, '0' is labelled as 10
        target[y_grid_coord, x_grid_coord, start:end] = [1, x / image_size[0], y / image_size[1],
                                                         width / image_size[0], height / image_size[1]] + label_dummy

    return target


def get_index_in_range(coordinate, length, splits):
    return sum(coordinate > i * length / splits for i in range(splits)) - 1


def get_anchor_box_index(anchor_boxes, box_size):
    """
    :param anchor_boxes: List of anchor boxes, i.e. [(10, 10), (10, 20), ...]
    :param box_size: Tuple of width, height of the box
    :return: The index of the most similar anchor box
    """
    IOUs = [intersection_over_union(anchor_box, box_size) for anchor_box in anchor_boxes]
    return [i for i, x in enumerate(IOUs) if x == max(IOUs)][0]


def intersection_over_union(box_1, box_2):
    """Assumes the boxes on centered on each other"""
    intersection_width = min(box_1[0], box_2[0])
    intersection_height = min(box_1[1], box_2[1])
    intersection = intersection_width * intersection_height
    union = box_1[0] * box_1[1] + box_2[0] * box_2[1] - intersection
    return intersection / union


if __name__ == '__main__':
    import cPickle as pickle

    with open('../../input/train_bounding_boxes.p') as f:
        d = pickle.load(f)


