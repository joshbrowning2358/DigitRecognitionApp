from __future__ import division
from unittest import TestCase

import numpy as np

import DigitRecognition.data.bounding_boxes_to_target as script


class TestBoundingBoxesToTarget(TestCase):
    def test_bounding_boxes_to_target_computes_correct_boxes_for_example(self):
        target = script.bounding_boxes_to_target(bounding_boxes=[(3, 31, 13, 10, 38), (0, 45, 15, 10, 38),
                                                                 (5, 59, 17, 10, 38), (5, 73, 19, 10, 38)],
                                                 anchor_boxes=[(10, 20), (20, 40)],
                                                 image_size=(100, 100),
                                                 grid_rows=10)

        # Is object?
        expected = np.zeros((10, 10))
        expected[3, 3] = 1
        expected[3, 4] = 1
        expected[3, 6] = 1
        expected[3, 7] = 1
        self.assertTrue(all(target[:, :, 0] == expected))

        # Object location (x)
        expected = np.zeros((10, 10))
        expected[3, 3] = .31
        expected[3, 4] = .45
        expected[3, 6] = .59
        expected[3, 7] = .73
        self.assertTrue(all(target[:, :, 1] == expected))

        # Object width
        expected = np.zeros((10, 10))
        expected[3, 3] = .38
        expected[3, 4] = .38
        expected[3, 6] = .38
        expected[3, 7] = .38
        self.assertTrue(all(target[:, :, 3] == expected))

    def test_intersection_over_union_for_simple_examples(self):
        result = script.intersection_over_union((10, 20), (10, 10))
        self.assertEqual(result, 0.5)

        result = script.intersection_over_union((20, 20), (30, 30))
        self.assertEqual(result, 4/9)

    def test_intersection_over_union_provides_same_result_for_random_boxes_but_switched_order(self):
        box_1 = np.random.randint(0, 100, size=2)
        box_2 = np.random.randint(0, 100, size=2)
        result1 = script.intersection_over_union(tuple(box_1), tuple(box_2))
        result2 = script.intersection_over_union(tuple(box_2), tuple(box_1))
        self.assertEqual(result1, result2)

    def test_anchor_boxes_chooses_correct_box_with_sample_data(self):
        anchor_boxes = [(10, 10), (20, 20), (30, 30)]
        index = script.get_anchor_box_index(anchor_boxes, (10, 10))
        self.assertEqual(index, 0)

        index = script.get_anchor_box_index(anchor_boxes, (10, 15))
        self.assertEqual(index, 0)

        index = script.get_anchor_box_index(anchor_boxes, (50, 50))
        self.assertEqual(index, 2)

    def test_get_index_in_range_works_correctly_with_sample_data(self):
        result = script.get_index_in_range(1, 10, 5)
        self.assertEqual(result, 0)

        result = script.get_index_in_range(5, 10, 5)
        self.assertEqual(result, 3)
