
import tensorflow as tf

def iou_train(boxA, boxB, realBox):
    """
    Calculate IoU between boxA and realBox
    Calculate the IoU in training phase, to get the box (out of N boxes per grid) responsible for ground truth box
    """
    iou1 = tf.reshape(iou_train_unit(boxA, realBox), [-1, 16, 9, 1])
    iou2 = tf.reshape(iou_train_unit(boxB, realBox), [-1, 16, 9, 1])
    return tf.concat([iou1, iou2], 3)

def iou_train_unit(boxA, realBox):
    """
    Calculate IoU between boxA and realBox
    """
    # make sure that the representation of box matches input
    intersectionX = tf.minimum(boxA[:, :, :, 0] + 0.5*boxA[:, :, :, 2], realBox[:, :, :, 0] + 0.5*realBox[:, :, :, 2]) - \
                    tf.maximum(boxA[:, :, :, 0] - 0.5*boxA[:, :, :, 2], realBox[:, :, :, 0] - 0.5*realBox[:, :, :, 2])
    intersectionY = tf.minimum(boxA[:, :, :, 1] + 0.5*boxA[:, :, :, 3], realBox[:, :, :, 1] + 0.5*realBox[:, :, :, 3]) - \
                    tf.maximum(boxA[:, :, :, 1] - 0.5*boxA[:, :, :, 3], realBox[:, :, :, 1] - 0.5*realBox[:, :, :, 3])
    intersection = tf.multiply(tf.maximum(0., intersectionX), tf.maximum(0., intersectionY))
    union = tf.subtract(tf.multiply(boxA[:, :, :, 1], boxA[:, :, :, 3]) + tf.multiply(realBox[:, :, :, 1], realBox[:, :, :, 3]), intersection)
    iou = tf.divide(intersection, union)
    return iou


def yolo_loss(pred, label, obj, no_obj):

    # shape(pred)   = [1, 16, 9, 10]
    # shape(label)  = [1, 16, 9, 5]
    # shape(obj)    = [1, 16, 9]
    # shape(no_obj) = [1, 16, 9]

    pred   = tf.reshape(pred,   [-1, 16, 9, 5, 2])

    ######################################

    pred_box1 = pred[:, :, :, 0:4, 0]
    pred_box2 = pred[:, :, :, 0:4, 1]
    label_box = label[:, :, :, 0:4]

    iou = iou_train(pred_box1, pred_box2, label_box)
    bb = tf.greater(iou[:, :, :, 0], iou[:, :, :, 1])
    return bb












