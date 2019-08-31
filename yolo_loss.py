
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

'''
def mean_sum_squared(loss):
    return tf.mean(tf.sum(tf.suared()))
'''

def yolo_loss(pred, label, obj, no_obj, cat):

    # pred   = [-1, 7, 7, 2 * 5 + 80]
    # label  = [-1, 7, 7, 5]
    # obj    = [-1, 7, 7]
    # no_obj = [-1, 7, 7]
    # cat    = [-1, 7, 7]

    ######################################

    label_box = label[:, :, :, 0:4]
    pred_box1 = pred[:, :, :, 0:4]
    pred_box2 = pred[:, :, :, 5:9]

    label_xy = label[:, :, :, 0:2]
    pred_xy1 = pred[:, :, :, 0:2]
    pred_xy2 = pred[:, :, :, 5:7]

    label_wh = tf.sqrt(label[:, :, :, 2:4])
    pred_wh1 = tf.sqrt(tf.abs(pred[:, :, :, 2:4])) * tf.sign(pred[:, :, :, 2:4])
    pred_wh2 = tf.sqrt(tf.abs(pred[:, :, :, 7:9])) * tf.sign(pred[:, :, :, 7:9])

    label_conf = label[:, :, :, 4]
    pred_conf1 = pred[:, :, :, 4]
    pred_conf2 = pred[:, :, :, 9]

    # we should only sigmoid the coordinate/confidence outputs
    label_cat = tf.one_hot(cat, depth=80)
    pred_cat = pred[:, :, :, 10:90]
    
    iou = iou_train(pred_box1, pred_box2, label_box)
    resp_box = tf.greater(iou[:, :, :, 0], iou[:, :, :, 1])

    ######################################

    # TODO
    # TP = (conf > thresh) and (pred_cat = label_cat) and (iou > 0.5)
    # whatever thresh we use for FP we must also use to TP.

    # https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
    # precision = TP / (TP + FP)

    threshold = tf.ones_like(pred_conf1) * 0.2
    conf_mask = tf.cast(tf.greater(tf.where(resp_box, pred_conf1, pred_conf2), threshold), tf.float32)

    TP = tf.count_nonzero(tf.greater(obj * tf.reduce_max(iou, axis=3) * conf_mask, 0.5 * tf.ones_like(obj)))
    TP_FN = tf.count_nonzero(obj)
    TP_FP = tf.count_nonzero(tf.greater(pred_conf1, threshold)) + tf.count_nonzero(tf.greater(pred_conf2, threshold))

    precision = tf.cast(TP, tf.float32) / (tf.cast(TP_FP, tf.float32) + 1e-3)
    recall = tf.cast(TP, tf.float32) / (tf.cast(TP_FN, tf.float32) + 1e-3)

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

    pred_cat = tf.reshape(obj, [-1,7,7,1]) * pred_cat
    cat_loss = tf.reduce_mean(tf.square(pred_cat - label_cat), axis=3)

    ######################################

    total_loss = xy_loss + wh_loss + obj_loss + no_obj_loss + cat_loss
    # total_loss = tf.Print(total_loss, [tf.shape(xy_loss), tf.shape(wh_loss), tf.shape(obj_loss), tf.shape(no_obj_loss), tf.shape(cat_loss)], message='', summarize=1000)

    loss = tf.reduce_mean(tf.reduce_sum(total_loss, axis=[1, 2]))

    return loss, precision, recall











