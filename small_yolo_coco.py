

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--name', type=str, default='yolo_coco')
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import keras
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=10000)
import cv2

from bc_utils.conv_utils import conv_output_length
from bc_utils.conv_utils import conv_input_length

from bc_utils.init_tensor import init_filters
from bc_utils.init_tensor import init_matrix

from LoadCOCO import LoadCOCO
from yolo_loss import yolo_loss
from draw_boxes import draw_boxes

from collections import deque

##############################################

def write(text):
    print (text)
    f = open(args.name + '.results', "a")
    f.write(text + "\n")
    f.close()

##############################################

loader = LoadCOCO()
weights = np.load('small_yolo_weights.npy', allow_pickle=True).item()

###############################################################

def max_pool(x, s):
    return tf.nn.max_pool(x, ksize=[1,s,s,1], strides=[1,s,s,1], padding='SAME')

def conv(x, f, p, w, name):
    fw, fh, fi, fo = f

    trainable = (w == None)

    if w is not None:
        print ('loading %s | trainable %d ' % (name, trainable))
        filters_np = w[name]
        bias_np    = w[name + '_bias']
    else:
        print ('making %s | trainable %d ' % (name, trainable))
        filters_np = init_filters(size=[fw, fh, fi, fo], init='glorot_uniform')
        bias_np    = np.zeros(shape=fo)

    if not (np.shape(filters_np) == f):
        print (np.shape(filters_np), f)
        assert(np.shape(filters_np) == f)

    filters = tf.Variable(filters_np, dtype=tf.float32, trainable=trainable)
    bias    = tf.Variable(bias_np,    dtype=tf.float32, trainable=trainable)

    conv = tf.nn.conv2d(x, filters, [1,p,p,1], 'SAME') + bias
    relu = tf.nn.leaky_relu(conv, 0.1)

    return relu

def dense(x, size, w, name):
    input_size, output_size = size

    trainable = (w == None)

    if w is not None:
        print ('loading %s | trainable %d ' % (name, trainable))
        weights_np = w[name]
        bias_np    = w[name + '_bias']
    else:
        print ('making %s | trainable %d ' % (name, trainable))
        weights_np = init_matrix(size=size, init='glorot_uniform')
        bias_np    = np.zeros(shape=output_size)

    w = tf.Variable(weights_np, dtype=tf.float32, trainable=trainable)
    b  = tf.Variable(bias_np, dtype=tf.float32, trainable=trainable)

    out = tf.matmul(x, w) + b
    return out

###############################################################

image_ph  = tf.placeholder(tf.float32, [1, 448, 448, 3])
coords_ph = tf.placeholder(tf.float32, [None, 7, 7, 5])
obj_ph    = tf.placeholder(tf.float32, [None, 7, 7])
no_obj_ph = tf.placeholder(tf.float32, [None, 7, 7])
cat_ph    = tf.placeholder(tf.int32,   [None, 7, 7])

lr_ph = tf.placeholder(tf.float32, ())

###############################################################

x = (image_ph / 255.0) * 2.0 - 1.0                                # 448

conv1 = conv(x, (7,7,3,64), 2, weights, 'conv_1')                 # 448
pool1 = max_pool(conv1, 2)                                        # 224
conv2 = conv(pool1, (3,3,64,192), 1, weights, 'conv_2')           # 112
pool2 = max_pool(conv2, 2)                                        # 112

conv3 = conv(pool2, (1,1,192,128), 1, weights, 'conv_3')          # 56
conv4 = conv(conv3, (3,3,128,256), 1, weights, 'conv_4')          # 56
conv5 = conv(conv4, (1,1,256,256), 1, weights, 'conv_5')          # 56
conv6 = conv(conv5, (3,3,256,512), 1, weights, 'conv_6')          # 56
pool3 = max_pool(conv6, 2)                                        # 56

conv7 = conv(pool3,   (1,1,512,256),  1, weights, 'conv_7')       # 28
conv8 = conv(conv7,   (3,3,256,512),  1, weights, 'conv_8')       # 28
conv9 = conv(conv8,   (1,1,512,256),  1, weights, 'conv_9')       # 28
conv10 = conv(conv9,  (3,3,256,512),  1, weights, 'conv_10')      # 28
conv11 = conv(conv10, (1,1,512,256),  1, weights, 'conv_11')      # 28
conv12 = conv(conv11, (3,3,256,512),  1, weights, 'conv_12')      # 28
conv13 = conv(conv12, (1,1,512,256),  1, weights, 'conv_13')      # 28
conv14 = conv(conv13, (3,3,256,512),  1, weights, 'conv_14')      # 28
conv15 = conv(conv14, (1,1,512,512),  1, weights, 'conv_15')      # 28
conv16 = conv(conv15, (3,3,512,1024), 1, weights, 'conv_16')      # 28
pool4 = max_pool(conv16, 2)                                       # 28

conv17 = conv(pool4,  (1,1,1024,512), 1, weights, 'conv_17')      # 14
conv18 = conv(conv17, (3,3,512,1024), 1, weights, 'conv_18')      # 14
conv19 = conv(conv18, (1,1,1024,512), 1, weights, 'conv_19')      # 14
conv20 = conv(conv19, (3,3,512,1024), 1, weights, 'conv_20')      # 14

conv21 = conv(conv20, (3,3,1024,1024), 1, None, 'conv_21')     # 14
conv22 = conv(conv21, (3,3,1024,1024), 2, None, 'conv_22')     # 14
conv23 = conv(conv22, (3,3,1024,1024), 1, None, 'conv_23')     # 7
conv24 = conv(conv23, (3,3,1024,1024), 1, None, 'conv_24')     # 7

flat = tf.reshape(conv24, [1, 7*7*1024])

dense1 = tf.nn.relu(dense(flat,   (7*7*1024,   4096), None, 'dense_1'))
dense2 =            dense(dense1, (    4096, 7*7*90), None, 'dense_2')

out = tf.reshape(dense2, [1, 7, 7, 90])

###############################################################

loss = yolo_loss(out, coords_ph, obj_ph, no_obj_ph, cat_ph)
train = tf.train.AdamOptimizer(learning_rate=lr_ph, epsilon=args.eps).minimize(loss)

###############################################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

###############################################################

counter = 0
losses = deque(maxlen=1000)
TPs = deque(maxlen=1000)
TP_FPs = deque(maxlen=1000)
TP_FNs = deque(maxlen=1000)

###############################################################

offset = [
[[0, 0], [0, 64], [0, 128], [0, 192], [0, 256], [0, 320], [0, 384]], 
[[64, 0], [64, 64], [64, 128], [64, 192], [64, 256], [64, 320], [64, 384]], 
[[128, 0], [128, 64], [128, 128], [128, 192], [128, 256], [128, 320], [128, 384]], 
[[192, 0], [192, 64], [192, 128], [192, 192], [192, 256], [192, 320], [192, 384]], 
[[256, 0], [256, 64], [256, 128], [256, 192], [256, 256], [256, 320], [256, 384]],  
[[320, 0], [320, 64], [320, 128], [320, 192], [320, 256], [320, 320], [320, 384]],  
[[384, 0], [384, 64], [384, 128], [384, 192], [384, 256], [384, 320], [384, 384]]
]

def calc_iou(label, pred1, pred2):
    iou1 = calc_iou_help(label, pred1)
    iou2 = calc_iou_help(label, pred2)
    return np.stack([iou1, iou2], 3)

def calc_iou_help(boxA, boxB):
    intersectionX = np.minimum(boxA[:, :, :, 0] + boxA[:, :, :, 2], boxB[:, :, :, 0] + boxB[:, :, :, 2]) - np.maximum(boxA[:, :, :, 0], boxB[:, :, :, 0])
    intersectionY = np.minimum(boxA[:, :, :, 1] + boxA[:, :, :, 3], boxB[:, :, :, 1] + boxB[:, :, :, 3]) - np.maximum(boxA[:, :, :, 1], boxB[:, :, :, 1])
    intersection = np.maximum(0., intersectionX) * np.maximum(0., intersectionY)
    union = (boxA[:, :, :, 2] * boxA[:, :, :, 3]) + (boxB[:, :, :, 2] * boxB[:, :, :, 3]) - intersection
    iou = intersection / union
    return iou

def grid_to_pix(box):
    box[:, :, :, 0:2] = 64.  * box[:, :, :, 0:2] + offset
    box[:, :, :, 2:4] = 448. * box[:, :, :, 2:4]
    return box

def mAP(label, pred, conf_thresh=0.2, iou_thresh=0.3):

    label = np.reshape(grid_to_pix(label[:, :, :, 0:5]), (-1, 7, 7, 5))
    pred1 = np.reshape(grid_to_pix(pred[:, :, :, 0:5]),   (1, 7, 7, 5))
    pred2 = np.reshape(grid_to_pix(pred[:, :, :, 5:10]),  (1, 7, 7, 5))

    iou = calc_iou(label, pred1, pred2)
    resp_box = iou[:, :, :, 0] < iou[:, :, :, 1]

    obj = label[:, :, :, 4]
    pred_conf1 = pred1[:, :, :, 4]
    pred_conf2 = pred2[:, :, :, 4]

    ###############################

    iou_mask = obj * np.max(iou, axis=3) > iou_thresh
    conf_mask = np.where(resp_box, np.ones_like(obj) * pred_conf1, np.ones_like(obj) * pred_conf2) > conf_thresh

    TP = np.count_nonzero(iou_mask * conf_mask)
    TP_FP = np.count_nonzero(obj)
    TP_FN = np.count_nonzero(pred_conf1 > conf_thresh) + np.count_nonzero(pred_conf2 > conf_thresh)

    return TP, TP_FP, TP_FN

###############################################################

while True:
    if not loader.empty():
        image, det = loader.pop()
        coords, obj, no_obj, cat = det

        if (np.any(coords < 0.) or np.any(coords > 1.1)):
            print (coords)
            assert(not (np.any(coords < 0.) or np.any(coords > 1.1)))
    
        lr = 1e-3 if counter < 50000 else 1e-2

        [out_np, loss_np, _] = sess.run([out, loss, train], feed_dict={image_ph: image, coords_ph: coords, obj_ph: obj, no_obj_ph: no_obj, cat_ph: cat, lr_ph: lr})

        assert(not np.any(np.isnan(image)))
        assert(not np.any(np.isnan(out_np)))

        losses.append(loss_np)
        counter = counter + 1

        ################################################

        TP, TP_FP, TP_FN = mAP(coords, out_np)

        TPs.append(TP)
        TP_FPs.append(TP_FP)
        TP_FNs.append(TP_FN)

        precision = np.sum(TPs) / (np.sum(TP_FPs) + 1e-3)
        recall = np.sum(TPs) / (np.sum(TP_FNs) + 1e-3)

        # print ('precision: %f | recall %f' % (precision, recall))

        ################################################

        if (counter % 100 == 0):
            # draw_boxes('%d.jpg' % (counter), image, out_np, det, iou_np)
            write("%d: lr %f loss %f precision %f recall %f" % (counter, lr, np.average(losses), precision, recall))

            '''
            test_vector = {}
            test_vector['image'] = image
            test_vector['predict'] = out_np
            test_vector['coords'] = coords
            test_vector['obj'] = obj
            test_vector['no_obj'] = no_obj
            test_vector['cat'] = cat
            test_vector['iou'] = iou_np
            np.save('test_vector_' + str(counter), test_vector)
            '''

###############################################################



    
    
    
    
    
    


