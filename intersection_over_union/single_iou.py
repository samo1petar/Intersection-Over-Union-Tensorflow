import numpy as np
import tensorflow as tf


def iou_np_single(singe_bbox: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    '''
    :param singe_bbox: [4]
    :param bboxes: [N, 4]
    :return: [N]

    Calculate singe_bbox iou with every box from bboxes. Something like -> iou(singe_bbox, bboxes[...])
    '''

    singe_bbox = np.tile(singe_bbox, bboxes.shape[0]).reshape(bboxes.shape[0], 4)

    first_points_mask  = singe_bbox[:, :2] >= bboxes[:, :2]
    second_points_mask = singe_bbox[:, 2:] <= bboxes[:, 2:]

    first_points = bboxes[:, :2].copy()
    first_points[first_points_mask] = singe_bbox[:, :2][first_points_mask]

    second_points = bboxes[:, 2:].copy()
    second_points[second_points_mask] = singe_bbox[:, 2:][second_points_mask]

    mid_bboxes = np.concatenate((first_points, second_points), axis=1)

    no_intersection_mask = np.logical_or(mid_bboxes[:, 0] > mid_bboxes[:, 2], mid_bboxes[:, 1] > mid_bboxes[:, 3])

    mid_bboxes_area = (mid_bboxes[:, 2] - mid_bboxes[:, 0]) * (mid_bboxes[:, 3] - mid_bboxes[:, 1])
    bbox_area = (singe_bbox[:, 2] - singe_bbox[:, 0]) * (singe_bbox[:, 3] - singe_bbox[:, 1])
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    iou = mid_bboxes_area / (bbox_area + bboxes_area - mid_bboxes_area)
    iou[no_intersection_mask] = 0

    return iou


def iou_tf_single(single_bbox: tf.Tensor, bboxes: tf.Tensor) -> tf.Tensor:
    '''
    :param single_bbox: [4]
    :param bboxes: [N, 4]
    :return: [N]

    Calculate singe_bbox iou with every box from bboxes. Something like -> iou(singe_bbox, bboxes[...])
    '''

    single_bbox = tf.reshape(tf.tile(single_bbox, [bboxes.shape[0]]), (bboxes.shape[0], 4))

    first_points_mask = single_bbox[:, :2] >= bboxes[:, :2]
    second_points_mask = single_bbox[:, 2:] <= bboxes[:, 2:]

    mask = tf.concat((first_points_mask, second_points_mask), axis=1)

    mid_bboxes = single_bbox * tf.cast(mask, single_bbox.dtype) + bboxes * tf.cast(tf.logical_not(mask), bboxes.dtype)

    no_intersection_mask = tf.logical_or(mid_bboxes[:, 0] > mid_bboxes[:, 2], mid_bboxes[:, 1] > mid_bboxes[:, 3])

    mid_bboxes_area = (mid_bboxes[:, 2] - mid_bboxes[:, 0]) * (mid_bboxes[:, 3] - mid_bboxes[:, 1])
    bbox_area = (single_bbox[:, 2] - single_bbox[:, 0]) * (single_bbox[:, 3] - single_bbox[:, 1])
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    iou = mid_bboxes_area / (bbox_area + bboxes_area - mid_bboxes_area)
    iou = iou * tf.cast(tf.logical_not(no_intersection_mask), iou.dtype)

    return iou