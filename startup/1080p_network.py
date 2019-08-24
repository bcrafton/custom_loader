

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--gpu', type=int, default=0)
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

from LoadMOT import LoadMOT
from yolo_loss import yolo_loss

##############################################

loader = LoadMOT()

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

def batch_norm(x, f):
    gamma = tf.Variable(np.ones(shape=f), dtype=tf.float32)
    beta = tf.Variable(np.zeros(shape=f), dtype=tf.float32)
    mean = tf.reduce_mean(x, axis=[0,1,2])
    _, var = tf.nn.moments(x - mean, axes=[0,1,2])
    bn = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-3)
    return bn

def block(x, f1, f2, p):
    filters = tf.Variable(init_filters(size=[7,7,f1,f2], init='alexnet'), dtype=tf.float32)
    conv = tf.nn.conv2d(x, filters, [1,p,p,1], 'SAME')
    bn   = batch_norm(conv, f2)
    relu = tf.nn.relu(bn)
    return relu

def mobile_block(x, f1, f2, p):
    filters1 = tf.Variable(init_filters(size=[4,4,f1,1], init='alexnet'), dtype=tf.float32)
    filters2 = tf.Variable(init_filters(size=[1,1,f1,f2], init='alexnet'), dtype=tf.float32)

    conv1 = tf.nn.depthwise_conv2d(x, filters1, [1,p,p,1], 'SAME')
    bn1   = batch_norm(conv1, f1)
    relu1 = tf.nn.relu(bn1)

    conv2 = tf.nn.conv2d(relu1, filters2, [1,1,1,1], 'SAME')
    bn2   = batch_norm(conv2, f2)
    relu2 = tf.nn.relu(bn2)

    return relu2

###############################################################

image_ph  = tf.placeholder(tf.float32, [1, 1920, 1080, 3])
coords_ph = tf.placeholder(tf.float32, [None, 16, 9, 5])
obj_ph    = tf.placeholder(tf.float32, [None, 16, 9])
no_obj_ph = tf.placeholder(tf.float32, [None, 16, 9])

bn     = batch_norm(image_ph, 3)                 # 1920

block1 = block(bn, 3, 32, 5)                     # 1920

block2 = mobile_block(block1, 32, 64, 1)         # 384
block3 = mobile_block(block2, 64, 128, 3)        # 384

block4 = mobile_block(block3, 128, 128, 1)       # 128
block5 = mobile_block(block4, 128, 256, 2)       # 128

block6 = mobile_block(block5, 256, 256, 1)       # 64
block7 = mobile_block(block6, 256, 512, 2)       # 64

block8 = mobile_block(block7, 512, 512, 1)       # 32
block9 = mobile_block(block8, 512, 512, 2)       # 32

# S × S × (B ∗ 5 + C) 
# C = 0.
# B = 2.
block10 = mobile_block(block9, 512, 10, 1)       # 16

###############################################################

loss = yolo_loss(block10, coords_ph, obj_ph, no_obj_ph)
train = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1.).minimize(loss)
params = tf.trainable_variables()

###############################################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

'''
[params] = sess.run([params], feed_dict={})
for p in params:
    print (np.shape(p))
assert (False)
'''

###############################################################

while True:
    if not loader.empty():
        image, (coords, obj, no_obj) = loader.pop()
        image = np.transpose(image, [1, 0, 2])
        image = np.reshape(image, [1, 1920, 1080, 3])

        [l, _] = sess.run([loss, train], feed_dict={image_ph: image, coords_ph: coords, obj_ph: obj, no_obj_ph: no_obj})
        # print (np.shape(l))
        print (l)

###############################################################



    
    
    
    
    
    


