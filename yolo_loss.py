
import numpy as np
import tensorflow as tf

offset_np = [
[[0, 0],   [0, 64],   [0, 128],   [0, 192],   [0, 256],   [0, 320],   [0, 384]], 
[[64, 0],  [64, 64],  [64, 128],  [64, 192],  [64, 256],  [64, 320],  [64, 384]], 
[[128, 0], [128, 64], [128, 128], [128, 192], [128, 256], [128, 320], [128, 384]], 
[[192, 0], [192, 64], [192, 128], [192, 192], [192, 256], [192, 320], [192, 384]], 
[[256, 0], [256, 64], [256, 128], [256, 192], [256, 256], [256, 320], [256, 384]],  
[[320, 0], [320, 64], [320, 128], [320, 192], [320, 256], [320, 320], [320, 384]],  
[[384, 0], [384, 64], [384, 128], [384, 192], [384, 256], [384, 320], [384, 384]]
]

offset = tf.constant(offset_np, dtype=tf.float32)

def grid_to_pix(box):
    pix_box_xy = 64. * box[:, :, :, 0:2] + offset
    pix_box_wh = 448. * box[:, :, :, 2:4]
    pix_box = tf.concat((pix_box_xy, pix_box_wh), axis=3)
    return pix_box

def calc_iou(boxA, boxB, realBox):
    iou1 = calc_iou_help(boxA, realBox)
    iou2 = calc_iou_help(boxB, realBox)
    return tf.stack([iou1, iou2], 3)

def calc_iou_help(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = tf.maximum(boxA[:,:,:,0], boxB[:,:,:,0])
    yA = tf.maximum(boxA[:,:,:,1], boxB[:,:,:,1])
    xB = tf.minimum(boxA[:,:,:,2], boxB[:,:,:,2])
    yB = tf.minimum(boxA[:,:,:,3], boxB[:,:,:,3])

    # compute the area of intersection rectangle
    ix = xB - xA + 1.
    iy = yB - yA + 1.
    interArea = tf.maximum(tf.zeros_like(ix), ix) * tf.maximum(tf.zeros_like(iy), iy)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[:,:,:,2] - boxA[:,:,:,0] + 1.) * (boxA[:,:,:,3] - boxA[:,:,:,1] + 1.)
    boxBArea = (boxB[:,:,:,2] - boxB[:,:,:,0] + 1.) * (boxB[:,:,:,3] - boxB[:,:,:,1] + 1.)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def yolo_loss(pred, label, obj, no_obj, cat):

    # pred   = [-1, 7, 7, 2 * 5 + 80]
    # label  = [-1, 7, 7, 5]
    # obj    = [-1, 7, 7]
    # no_obj = [-1, 7, 7]
    # cat    = [-1, 7, 7]

    ######################################

    label_box = grid_to_pix(label[:, :, :, 0:4])
    pred_box1 = grid_to_pix(tf.nn.relu(pred[:, :, :, 0:4]))
    pred_box2 = grid_to_pix(tf.nn.relu(pred[:, :, :, 5:9]))

    ############################

    label_xy = label[:, :, :, 0:2]
    pred_xy1 = pred[:, :, :, 0:2]
    pred_xy2 = pred[:, :, :, 5:7]

    ############################

    label_wh = tf.sqrt(label[:, :, :, 2:4])
    pred_wh1 = tf.sqrt(tf.abs(pred[:, :, :, 2:4])) * tf.sign(pred[:, :, :, 2:4])
    pred_wh2 = tf.sqrt(tf.abs(pred[:, :, :, 7:9])) * tf.sign(pred[:, :, :, 7:9])

    ############################

    label_conf = label[:, :, :, 4]
    pred_conf1 = pred[:, :, :, 4]
    pred_conf2 = pred[:, :, :, 9]

    label_cat = tf.one_hot(cat, depth=80)
    pred_cat = pred[:, :, :, 10:90]
    
    iou = calc_iou(pred_box1, pred_box2, label_box)
    resp_box = tf.greater(iou[:, :, :, 0], iou[:, :, :, 1])

    ######################################

    threshold = tf.ones_like(pred_conf1) * 0.1
    conf_mask = tf.cast(tf.greater(tf.where(resp_box, tf.ones_like(obj) * pred_conf1, tf.ones_like(obj) * pred_conf2), threshold), tf.float32)

    TP = tf.count_nonzero(tf.greater(obj * tf.reduce_max(iou, axis=3), 0.5 * tf.ones_like(obj)))
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

    total_loss = xy_loss + wh_loss + obj_loss + no_obj_loss # + cat_loss
    # total_loss = tf.Print(total_loss, [tf.shape(label_box), tf.math.reduce_std(label_box), tf.shape(pred_box1), tf.math.reduce_std(pred_box1)], message='', summarize=1000)

    loss = tf.reduce_mean(tf.reduce_sum(total_loss, axis=[1, 2]))

    return loss, precision, recall, iou











