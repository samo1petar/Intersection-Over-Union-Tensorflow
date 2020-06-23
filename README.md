# Intersection-Over-Union-Tensorflow

This is a tensorflow implementation of intersection over union (iou). <br>
IOU is a measure of bounding box correctness used in detection. It's defines as 
bounding box intersection area divided with union area of both bounding boxes.

![Alt text](images/iou_equation.png?raw=true "IOU")

Main contribution of this repository is it's usability for neural networks out of the box.
The function _iou_tf()_ located inside _intersection_over_union/iou.py_ calculates
intersections of every proposal with every anchor. _iou_tf()_ takes proposals and anchors, and outputs iou values.
It's made so it takes feature map directly from some neural network layer (e.g. conv) in 
format [batch, height, width, anchor, points], and outputs values in format [batch, height, width, anchor, iou].
In this way it's values are easily interpretable.

iou_tf()
 - input: proposals -> [batch, height, width, anchor, points]
 - input: anchors -> [N, points]
 - output: iou values -> [batch, height, width, anchor, iou]

Algorithm speed depends upon anchor number and feature map size.
Next graph shows algorithm speed on Nvidia GTX 1070. 
In most detection cases anchor number is usually lower than 20, which means this algorithm is fast for all
but for big feature maps (100x100) with a lot of different anchors. But even than, it works great for up to ~25 anchors.

On the image bellow legend represents feature map size.
![Alt text](images/Times_1.png?raw=true "IOU speed")
