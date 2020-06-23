import tensorflow as tf


def iou_tf(proposals: tf.Tensor, anchors: tf.Tensor) -> tf.Tensor:

    '''
    :param proposals: [batch, height, width, anchor, points]
    :param anchors: [N, points]
    :return: [batch, height, width, anchor, iou]

    proposals are proposals that come from feature_map, while anchors are anchors
    '''

    proposals_tiled = tf.tile(proposals, [1, 1, 1, anchors.shape[0], 1])
    proposals_tiled_sh = tf.reshape(proposals_tiled,
                                       [proposals_tiled.shape[0],
                                        proposals_tiled.shape[1],
                                        proposals_tiled.shape[2],
                                        anchors.shape[0],
                                        proposals.shape[3],
                                        proposals_tiled.shape[4],
                                        ])
    anchors_tiled = tf.tile(anchors, [1, proposals_tiled_sh.shape[4]])
    anchors_tiled_sh = tf.reshape(anchors_tiled,
                                  [anchors.shape[0],
                                   proposals_tiled_sh.shape[4],
                                   anchors.shape[1],
                                   ])

    first_points_mask  = proposals_tiled_sh[:, :, :, :, :, :2] >= anchors_tiled_sh[:, :, :2]
    second_points_mask = proposals_tiled_sh[:, :, :, :, :, 2:] <= anchors_tiled_sh[:, :, 2:]

    mask = tf.concat((first_points_mask, second_points_mask), axis=-1)

    mid_bboxes = proposals_tiled_sh * tf.cast(mask, proposals_tiled_sh.dtype) + \
                 anchors_tiled_sh * tf.cast(tf.logical_not(mask), anchors_tiled_sh.dtype)

    no_intersection_mask = tf.logical_or(mid_bboxes[..., 0] > mid_bboxes[..., 2], mid_bboxes[..., 1] > mid_bboxes[..., 3])

    mid_bboxes_area = (mid_bboxes[..., 2] - mid_bboxes[..., 0]) * (mid_bboxes[..., 3] - mid_bboxes[..., 1])
    proposals_area = (proposals_tiled_sh[..., 2] - proposals_tiled_sh[..., 0]) * (proposals_tiled_sh[..., 3] - proposals_tiled_sh[..., 1])
    anchors_area = (anchors_tiled_sh[..., 2] - anchors_tiled_sh[..., 0]) * (anchors_tiled_sh[..., 3] - anchors_tiled_sh[..., 1])

    iou = mid_bboxes_area / (proposals_area + anchors_area - mid_bboxes_area)
    iou = iou * tf.cast(tf.logical_not(no_intersection_mask), iou.dtype)

    return iou
