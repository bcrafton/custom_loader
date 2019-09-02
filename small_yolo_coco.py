

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

##############################################

def in_top_k(x, y, k):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)

    _, topk = tf.nn.top_k(input=x, k=k)
    topk = tf.transpose(topk)
    correct = tf.equal(y, topk)
    correct = tf.cast(correct, dtype=tf.int32)
    correct = tf.reduce_sum(correct, axis=0)
    return correct

###############################################################

def max_pool(x, s):
    return tf.nn.max_pool(x, ksize=[1,s,s,1], strides=[1,s,s,1], padding='SAME')

def batch_norm(x, f, name):
    gamma = tf.Variable(np.ones(shape=f), dtype=tf.float32, name=name+'_gamma')
    beta = tf.Variable(np.zeros(shape=f), dtype=tf.float32, name=name+'_beta')
    mean = tf.reduce_mean(x, axis=[0,1,2])
    _, var = tf.nn.moments(x - mean, axes=[0,1,2])
    bn = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-3)
    return bn

def conv(x, f, p, w, name):
    fw, fh, fi, fo = f

    if w:
        filters_np = w[name]
        bias_np    = w[name + '_bias']
    else:
        filters_np = init_filters(size=[fw, fh, fi, fo], init='alexnet')
        bias_np    = np.zeros(shape=fo)

    if not (np.shape(filters_np) == f):
        print (np.shape(filters_np), f)
        assert(np.shape(filters_np) == f)

    filters = tf.Variable(filters_np, dtype=tf.float32, trainable=False)
    bias    = tf.Variable(bias_np,    dtype=tf.float32, trainable=False)

    conv = tf.nn.conv2d(x, filters, [1,p,p,1], 'SAME') + bias
    relu = tf.nn.relu(conv)

    return relu

###############################################################

image_ph  = tf.placeholder(tf.float32, [1, 448, 448, 3])
coords_ph = tf.placeholder(tf.float32, [None, 7, 7, 5])
obj_ph    = tf.placeholder(tf.float32, [None, 7, 7])
no_obj_ph = tf.placeholder(tf.float32, [None, 7, 7])
cat_ph    = tf.placeholder(tf.int32, [None, 7, 7])

###############################################################

# x = image_ph
# x = batch_norm(image_ph, 3, 'bn0')                              # 448
x = (image_ph / 255.0) * 2.0 - 1.0

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

conv21 = conv(conv20, (3,3,1024,1024), 1, weights, 'conv_21')     # 14
conv22 = conv(conv21, (3,3,1024,1024), 2, weights, 'conv_22')     # 14
conv23 = conv(conv22, (3,3,1024,1024), 1, weights, 'conv_23')     # 7
conv24 = conv(conv23, (3,3,1024,1024), 1, weights, 'conv_24')     # 7

###############################################################

'''
mat1   = tf.Variable(weights['dense_1'], dtype=tf.float32)
bias1  = tf.Variable(weights['dense_1_bias'], dtype=tf.float32)
mat2   = tf.Variable(weights['dense_2'], dtype=tf.float32)
bias2  = tf.Variable(weights['dense_2_bias'], dtype=tf.float32)
mat3   = tf.Variable(weights['dense_3'], dtype=tf.float32)
bias3  = tf.Variable(weights['dense_3_bias'], dtype=tf.float32)
'''

mat1   = tf.Variable(init_matrix(size=(7*7*1024, 512), init='glorot_normal'), dtype=tf.float32)
bias1  = tf.Variable(np.zeros(shape=512), dtype=tf.float32)
mat2   = tf.Variable(init_matrix(size=(512, 4096), init='glorot_normal'), dtype=tf.float32)
bias2  = tf.Variable(np.zeros(shape=4096), dtype=tf.float32)
mat3   = tf.Variable(init_matrix(size=(4096, 7*7*90), init='glorot_normal'), dtype=tf.float32)
bias3  = tf.Variable(np.zeros(shape=7*7*90), dtype=tf.float32)

flat   = tf.reshape(conv24, [1, 7*7*1024])

dense1 = tf.matmul(flat,  mat1) + bias1
relu1  = tf.nn.relu(dense1)

dense2 = tf.matmul(relu1, mat2) + bias2
relu2  = tf.nn.relu(dense2)

dense3 = tf.matmul(relu2, mat3) + bias3
out    = tf.reshape(dense3, [1, 7, 7, 90])

###############################################################

loss, precision, recall, iou = yolo_loss(out, coords_ph, obj_ph, no_obj_ph, cat_ph)
train = tf.train.AdamOptimizer(learning_rate=args.lr, epsilon=args.eps).minimize(loss)

###############################################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

###############################################################

counter = 0
losses = deque(maxlen=10000)
precs = deque(maxlen=10000)
recs = deque(maxlen=10000)

while True:
    if not loader.empty():
        image, det = loader.pop()
        coords, obj, no_obj, cat = det

        if (np.any(coords < 0.) or np.any(coords > 1.1)):
            print (coords)
            assert(not (np.any(coords < 0.) or np.any(coords > 1.1)))

        [p, i, l, prec, rec, _] = sess.run([out, iou, loss, precision, recall, train], feed_dict={image_ph: image, coords_ph: coords, obj_ph: obj, no_obj_ph: no_obj, cat_ph: cat})

        losses.append(l)
        precs.append(prec)
        recs.append(rec)
        counter = counter + 1

        if (counter % 1000 == 0):
            draw_boxes('%d.jpg' % (counter), image, p, det, i)
            write("%d: %f %f %f" % (counter, np.average(losses), np.average(precs), np.average(recs)))

###############################################################



    
    
    
    
    
    


