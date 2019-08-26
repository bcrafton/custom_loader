
import tensorflow as tf

def iou_train(boxA, boxB, realBox):
    """
    Calculate IoU between boxA and realBox
    Calculate the IoU in training phase, to get the box (out of N boxes per grid) responsible for ground truth box
    """
    iou1 = tf.reshape(iou_train_unit(boxA, realBox), [-1, 7, 7, 1])
    iou2 = tf.reshape(iou_train_unit(boxB, realBox), [-1, 7, 7, 1])
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

    pred   = tf.reshape(pred,   [-1, 7, 7, 2, 5])

    ######################################

    label_box = label[:, :, :, 0:4]
    pred_box1 = pred[:, :, :, 0, 0:4]
    pred_box2 = pred[:, :, :, 1, 0:4]

    label_xy = label[:, :, :, 0:2]
    pred_xy1 = pred[:, :, :, 0, 0:2]
    pred_xy2 = pred[:, :, :, 1, 0:2]

    label_wh = tf.sqrt(label[:, :, :, 2:4])
    pred_wh1 = tf.sqrt(pred[:, :, :, 0, 2:4])
    pred_wh2 = tf.sqrt(pred[:, :, :, 1, 2:4])

    label_conf = label[:, :, :, 4]
    pred_conf1 = pred[:, :, :, 0, 4]
    pred_conf2 = pred[:, :, :, 1, 4]

    iou = iou_train(pred_box1, pred_box2, label_box)
    resp_box = tf.greater(iou[:, :, :, 0], iou[:, :, :, 1])

    ######################################

    # https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
    # precision = TP / (TP + FP)
    correct = tf.count_nonzero(tf.greater(obj * tf.reduce_max(iou, axis=3), 0.5 * tf.ones_like(obj)))
    total = tf.count_nonzero(obj)
    mAP = tf.cast(correct, tf.float32) / tf.cast(total, tf.float32)

    ######################################

    loss_xy1 = tf.reduce_sum(tf.square(pred_xy1 - label_xy), 3)
    loss_xy2 = tf.reduce_sum(tf.square(pred_xy2 - label_xy), 3)
    xy_loss = 5. * obj * tf.where(resp_box, loss_xy1, loss_xy2)

    ######################################

    loss_wh1 = tf.reduce_sum(tf.square(pred_wh1 - label_wh), 3)
    loss_wh2 = tf.reduce_sum(tf.square(pred_wh2 - label_wh), 3)
    wh_loss = 5. * obj * tf.where(resp_box, loss_wh1, loss_wh2)

    ######################################

    loss_obj1 = tf.square(pred_conf1 - label_conf)
    loss_obj2 = tf.square(pred_conf2 - label_conf)
    obj_loss = 1. * obj * tf.where(resp_box, loss_obj1, loss_obj2)

    ######################################    

    loss_no_obj1 = tf.square(pred_conf1 - label_conf)
    loss_no_obj2 = tf.square(pred_conf2 - label_conf)
    no_obj_loss = 0.5 * no_obj * tf.where(resp_box, loss_no_obj1, loss_no_obj2)

    ######################################

    total_loss = xy_loss + wh_loss + obj_loss + no_obj_loss # [?, 16, 9]
    loss = tf.reduce_mean(tf.reduce_sum(total_loss, axis=[1, 2]))

    return loss, mAP











