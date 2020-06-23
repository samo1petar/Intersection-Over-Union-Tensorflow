import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import time
from typing import List

from intersection_over_union.iou import iou_tf
from intersection_over_union.single_iou import iou_np_single, iou_tf_single



def test_single_example():

    '''
    Test intersection over union on the data bellow.

    gt_bbox shape -> [N, 4]
    out_bbox shape -> [batch, height, width, anchor, points]

    Here height and width are simulated as feature extractor final layer. Bounding box positions
    are kept intact, so they are easily interpreted. In this way in can be used directly in
    neural network without any modification.
    In this example, it's set up as 2 x 4, but it can be any number. If you want to test on
    different example, adapt out_bbox accordingly.
    '''

    gt_bbox = np.array([
        [10, 10, 20, 20],
        [25, 25, 30, 30],
    ])

    out_bbox = np.array([
        [5, 15, 15, 25],
        [12, 5, 18, 25],
        [5, 5, 12, 12],
        [8, 8, 15, 15],
        [10, 10, 20, 20],
        [16, 16, 30, 30],
        [100, 100, 200, 200],
        [0, 0, 2, 2],
    ])

    rez_single_0 = iou_np_single(gt_bbox[0], out_bbox)
    rez_single_1 = iou_np_single(gt_bbox[1], out_bbox)

    out_bbox = tf.reshape(tf.convert_to_tensor(out_bbox, dtype=tf.float32), [1, 1, 4, 2, 4])
    gt_bbox = tf.convert_to_tensor(gt_bbox, dtype=tf.float32)
    rez_iou = iou_tf(out_bbox, gt_bbox).numpy()

    print ('Iou result from iou_tf():\n{}'.format(rez_iou[..., 0, :].reshape(-1)))
    print ('Iou result from np:\n {}'.format(rez_single_0))

    print ('Iou result from iou_tf():\n{}'.format(rez_iou[..., 1, :].reshape(-1)))
    print ('Iou result from np:\n {}'.format(rez_single_1))



def test_10k_random_examples():

    '''
    Test iou_tf on 10000 cases with random ground truth anchors and random proposed anchors.

    gt_bbox shape -> [N, 4]
    out_bbox shape -> [batch, height, width, anchor, points]

    Here height and width are simulated as feature extractor final layer. Bounding box positions
    are kept intact, so they are easily interpreted. In this way in can be used directly in
    neural network without any modification.
    In this example, it's set up as 2 x 4, but it can be any number. If you want to test on
    different example, adapt out_bbox accordingly.
    '''

    error_n = 0
    for n in range(10000):
        print (' {} / {}'.format(n, 10000), end='\r')

        n_anchors = np.random.randint(1, 20) # number of anchors
        f_height  = np.random.randint(1, 30) # feature_extractor height
        f_width   = np.random.randint(1, 30) # feature_extractor width

        gt_bboxes  = np.concatenate((np.random.uniform( 0,  50, 2 * n_anchors                     ).reshape(-1, 2),
                                     np.random.uniform(50, 100, 2 * n_anchors                     ).reshape(-1, 2)), axis=1)
        out_bboxes = np.concatenate((np.random.uniform( 0,  70, 2 * n_anchors * f_height * f_width).reshape(-1, 2),
                                     np.random.uniform(50, 100, 2 * n_anchors * f_height * f_width).reshape(-1, 2)), axis=1)

        out_bboxes_tf = tf.reshape(tf.convert_to_tensor(out_bboxes, dtype=tf.float32), [1, f_height, f_width, n_anchors, 4])
        gt_bboxes_tf = tf.convert_to_tensor(gt_bboxes, dtype=tf.float32)

        rez_iou_tf = iou_tf(out_bboxes_tf, gt_bboxes_tf).numpy()

        rez_iou_np = np.array([])
        for x in range(n_anchors):
            rez_np = iou_np_single(gt_bboxes[x], out_bboxes)
            rez_iou_np = np.append(rez_iou_np, rez_np).reshape(x + 1, -1)

        max_errors = []
        avg_errors = []
        for x in range(n_anchors):
            errors = np.abs(rez_iou_tf[..., x, :].reshape(-1) - rez_iou_np[x])
            max_errors.append(np.max(errors))
            avg_errors.append(np.average(errors))

        if max(max_errors) > 0.00001:
            print ('error found')
            error_n += 1
            print ()
    print ('test_10k_random_examples: {} errors found'.format(error_n))



def test_10k_on_single_bbox_functions():

    '''
    Compare iou_np_single and iou_tf_single functions.
    These functions are made for iou_tf testing.
    '''

    error_n = 0
    for n in range(10000):
        print (n, end='\r')

        bbox = np.concatenate((np.random.uniform(0, 50, 2), np.random.uniform(50, 100, 2)))
        bboxes = np.concatenate((np.random.uniform(0, 70, 2 * 1000).reshape(-1, 2), np.random.uniform(50, 100, 2 * 1000).reshape(-1, 2)), axis=1)

        np_output = iou_np_single(bbox, bboxes)
        tf_output = iou_tf_single(tf.constant(bbox), tf.constant(bboxes)).numpy()

        if np.sum(np.abs(np_output - tf_output)) > 0.000001:
            print ('error found')
            error_n += 1
            print ()
    print('test_10k_on_single_bbox_functions: {} errors found'.format(error_n))



def plot_speed():

    '''
    Plot average speed values with different parameters. Feature map size and anchor number are
    parameters that change in this test. Results are shown in README.md
    '''

    def test_speed(f_height: int, f_width: int, iterations: int):
        times = []
        for n in range(iterations + 1):
            print(n, end='\r')

            n_anchors = n  # number of anchors

            gt_bboxes =  np.concatenate((np.random.uniform( 0,  50, 2 * n_anchors                     ).reshape(-1, 2),
                                         np.random.uniform(50, 100, 2 * n_anchors                     ).reshape(-1, 2)), axis=1)
            out_bboxes = np.concatenate((np.random.uniform( 0,  70, 2 * n_anchors * f_height * f_width).reshape(-1, 2),
                                         np.random.uniform(50, 100, 2 * n_anchors * f_height * f_width).reshape(-1, 2)), axis=1)

            out_bboxes_tf = tf.reshape(tf.convert_to_tensor(out_bboxes, dtype=tf.float32),
                                       [1, f_height, f_width, n_anchors, 4])
            gt_bboxes_tf = tf.convert_to_tensor(gt_bboxes, dtype=tf.float32)

            start = time.time()
            rez_iou_tf = iou_tf(out_bboxes_tf, gt_bboxes_tf)
            end = time.time()
            rez_iou_tf = rez_iou_tf.numpy()
            times.append(end - start)
        return times

    def smooth(f_height: int, f_width: int, iterations: int, smooth: int = 10) -> List[float]:
        results = []
        for x in range(smooth):
            results.append(test_speed(f_height, f_width, iterations))
        results = np.array(results)
        return np.sum(results, axis=0) / smooth

    max_anchors = 200
    times_1x1 = smooth(1, 1, max_anchors)
    times_2x2 = smooth(2, 2, max_anchors)
    times_5x5 = smooth(5, 5, max_anchors)
    times_10x10 = smooth(10, 10, max_anchors)
    times_20x20 = smooth(20, 20, max_anchors)
    times_30x30 = smooth(30, 30, max_anchors)
    times_50x50 = smooth(50, 50, 120)
    times_100x100 = smooth(100, 100, 60)

    plt.plot(list(range(max_anchors)), times_1x1[1:], label='1x1')
    plt.plot(list(range(max_anchors)), times_2x2[1:], label='2x2')
    plt.plot(list(range(max_anchors)), times_5x5[1:], label='5x5')
    plt.plot(list(range(max_anchors)), times_10x10[1:], label='10x10')
    plt.plot(list(range(max_anchors)), times_20x20[1:], label='20x20')
    plt.plot(list(range(max_anchors)), times_30x30[1:], label='30x30')
    plt.plot(list(range(120)), times_50x50[1:], label='50x50')
    plt.plot(list(range(60)), times_100x100[1:], label='100x100')

    plt.xlabel('# anchors')
    plt.ylabel('seconds')
    plt.legend()
    plt.show()



def test_single_time():

    '''
    Testing the speed of one pass. 10 passes are done and median time value is printed.
    Change n_anchors, f_height and f_width params for testing.
    '''

    n_anchors = 30
    f_height  = 20
    f_width   = 20

    gt_bboxes =  np.concatenate((np.random.uniform( 0,  50, 2 * n_anchors                     ).reshape(-1, 2),
                                 np.random.uniform(50, 100, 2 * n_anchors                     ).reshape(-1, 2)), axis=1)
    out_bboxes = np.concatenate((np.random.uniform( 0,  70, 2 * n_anchors * f_height * f_width).reshape(-1, 2),
                                 np.random.uniform(50, 100, 2 * n_anchors * f_height * f_width).reshape(-1, 2)), axis=1)

    out_bboxes_tf = tf.reshape(tf.convert_to_tensor(out_bboxes, dtype=tf.float32),
                               [1, f_height, f_width, n_anchors, 4])
    gt_bboxes_tf = tf.convert_to_tensor(gt_bboxes, dtype=tf.float32)

    times = []
    for x in range(10):
        start = time.time()
        iou_tf(out_bboxes_tf, gt_bboxes_tf)
        end = time.time()
        times.append(end - start)

    times = np.array(times)
    print ('test_single_time: \nn_anchors {}\n feature_map_height {}\nfeature_map_width {}\ntime {}'.format(
        n_anchors, f_height, f_width, np.median(times)))



if __name__ == '__main__':

    test_single_example()
    test_10k_random_examples()
    plot_speed()
    test_single_time()