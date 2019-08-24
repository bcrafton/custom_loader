

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
parser.add_argument('--load', type=str, default='MobileNet224_weights.npy')
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

from collections import deque

##############################################

loader = LoadCOCO()
wd = np.load(args.load, allow_pickle=True).item()

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

def avg_pool(x, s):
    return tf.nn.avg_pool(bn, ksize=[1,s,s,1], strides=[1,s,s,1], padding='SAME')

def batch_norm(x, f, name, load):
    if load:
        gamma = tf.Variable(load[name+'_gamma'+':0'], dtype=tf.float32, name=name+'_gamma', trainable=False)
        beta = tf.Variable(load[name+'_beta'+':0'], dtype=tf.float32, name=name+'_beta', trainable=False)
    else:
        gamma = tf.Variable(np.ones(shape=f), dtype=tf.float32)
        beta = tf.Variable(np.zeros(shape=f), dtype=tf.float32)

    mean = tf.reduce_mean(x, axis=[0,1,2])
    _, var = tf.nn.moments(x - mean, axes=[0,1,2])
    bn = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-3)
    return bn

def block(x, f1, f2, p, name, load):
    if load:
        filters = tf.Variable(load[name+'_conv'+':0'], dtype=tf.float32, name=name+'_conv', trainable=False)
    else:
        filters = tf.Variable(init_filters(size=[3,3,f1,f2], init='alexnet'), dtype=tf.float32, name=name+'_conv')

    conv = tf.nn.conv2d(x, filters, [1,p,p,1], 'SAME')
    bn   = batch_norm(conv, f2, name+'_bn', load)
    relu = tf.nn.relu(bn)
    return relu

def mobile_block(x, f1, f2, p, name, load):
    print (name)
    if load:
        filters1 = tf.Variable(load[name+'_conv_dw'+':0'], dtype=tf.float32, name=name+'_conv_dw', trainable=False)
        filters2 = tf.Variable(load[name+'_conv_pw'+':0'], dtype=tf.float32, name=name+'_conv_pw', trainable=False)
    else:
        filters1 = tf.Variable(init_filters(size=[3,3,f1,1], init='alexnet'), dtype=tf.float32)
        filters2 = tf.Variable(init_filters(size=[1,1,f1,f2], init='alexnet'), dtype=tf.float32)

    conv1 = tf.nn.depthwise_conv2d(x, filters1, [1,p,p,1], 'SAME')
    bn1   = batch_norm(conv1, f1, name+'_bn_dw', load)
    relu1 = tf.nn.relu(bn1)

    conv2 = tf.nn.conv2d(relu1, filters2, [1,1,1,1], 'SAME')
    bn2   = batch_norm(conv2, f2, name+'_bn_pw', load)
    relu2 = tf.nn.relu(bn2)

    return relu2

###############################################################

image_ph  = tf.placeholder(tf.float32, [1, 448, 448, 3])
coords_ph = tf.placeholder(tf.float32, [None, 7, 7, 5])
obj_ph    = tf.placeholder(tf.float32, [None, 7, 7])
no_obj_ph = tf.placeholder(tf.float32, [None, 7, 7])

bn     = batch_norm(image_ph, 3, 'bn0', None)                 # 224 448

block1 = block(bn, 3, 32, 2, 'block1', wd)                    # 224 448

block2 = mobile_block(block1, 32, 64, 1, 'block2', wd)        # 112 224
block3 = mobile_block(block2, 64, 128, 2, 'block3', wd)       # 112 224

block4 = mobile_block(block3, 128, 128, 1, 'block4', wd)      # 56  112
block5 = mobile_block(block4, 128, 256, 2, 'block5', wd)      # 56  112

block6 = mobile_block(block5, 256, 256, 1, 'block6', wd)      # 28  56
block7 = mobile_block(block6, 256, 512, 2, 'block7', wd)      # 28  56

block8 = mobile_block(block7, 512, 512, 1, 'block8', wd)      # 14  28
block9 = mobile_block(block8, 512, 512, 2, 'block9', wd)      # 14  28
block10 = mobile_block(block9, 512, 512, 1, 'block10', wd)    # 14  14
block11 = mobile_block(block10, 512, 512, 1, 'block11', wd)   # 14  14
block12 = mobile_block(block11, 512, 512, 1, 'block12', wd)   # 14  14

block13 = mobile_block(block12, 512, 512, 2, 'block13', None) #     7
block14 = mobile_block(block13, 512, 512, 1, 'block14', None) #     7

block15 = mobile_block(block14, 512, 10, 1, 'block15', None) #     7
out = block15

'''
flat   = tf.reshape(block14, [1, 7*7*512])

mat1   = tf.Variable(init_matrix(size=(7*7*512, 4096), init='glorot_uniform'), dtype=tf.float32, name='fc1')
bias1  = tf.Variable(np.zeros(shape=4096), dtype=tf.float32, name='fc1_bias')
fc1    = tf.matmul(flat, mat1) + bias1
relu1  = tf.nn.relu(fc1)

mat2   = tf.Variable(init_matrix(size=(4096, 7*7*10), init='glorot_uniform'), dtype=tf.float32, name='fc2')
bias2  = tf.Variable(np.zeros(shape=7*7*10), dtype=tf.float32, name='fc2_bias')
fc2    = tf.matmul(relu1, mat2) + bias2

out    = tf.reshape(fc2, [1, 7, 7, 10])
'''


###############################################################

loss = yolo_loss(out, coords_ph, obj_ph, no_obj_ph)
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

counter = 0
losses = deque(maxlen=100)

while True:
    if not loader.empty():
        image, (coords, obj, no_obj) = loader.pop()

        [p, l, _] = sess.run([out, loss, train], feed_dict={image_ph: image, coords_ph: coords, obj_ph: obj, no_obj_ph: no_obj})
        losses.append(l)
        counter = counter + 1
        if (counter % 100 == 0):
            print (np.average(losses))

###############################################################



    
    
    
    
    
    


