

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

from LoadMOT import LoadMOT
from yolo_loss import yolo_loss

##############################################

loader = LoadMOT()
weight_dict = np.load(args.load, allow_pickle=True).item()

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

def batch_norm(x, f, name):
    gamma = tf.Variable(weight_dict[name+'_gamma'+':0'], dtype=tf.float32, name=name+'_gamma')
    beta = tf.Variable(weight_dict[name+'_beta'+':0'], dtype=tf.float32, name=name+'_beta')
    # gamma = tf.Variable(np.ones(shape=f), dtype=tf.float32)
    # beta = tf.Variable(np.zeros(shape=f), dtype=tf.float32)

    mean = tf.reduce_mean(x, axis=[0,1,2])
    _, var = tf.nn.moments(x - mean, axes=[0,1,2])
    bn = tf.nn.batch_normalization(x=x, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=1e-3)
    return bn

def block(x, f1, f2, p, name):
    filters = tf.Variable(weight_dict[name+'_conv'+':0'], dtype=tf.float32, name=name+'_conv')

    conv = tf.nn.conv2d(x, filters, [1,p,p,1], 'SAME')
    bn   = batch_norm(conv, f2, name+'_bn')
    relu = tf.nn.relu(bn)
    return relu

def mobile_block(x, f1, f2, p, name):
    filters1 = tf.Variable(weight_dict[name+'_conv_dw'+':0'], dtype=tf.float32, name=name+'_conv_dw')
    filters2 = tf.Variable(weight_dict[name+'_conv_pw'+':0'], dtype=tf.float32, name=name+'_conv_pw')

    conv1 = tf.nn.depthwise_conv2d(x, filters1, [1,p,p,1], 'SAME')
    bn1   = batch_norm(conv1, f1, name+'_bn_dw')
    relu1 = tf.nn.relu(bn1)

    conv2 = tf.nn.conv2d(relu1, filters2, [1,1,1,1], 'SAME')
    bn2   = batch_norm(conv2, f2, name+'_bn_pw')
    relu2 = tf.nn.relu(bn2)

    return relu2

###############################################################

image_ph  = tf.placeholder(tf.float32, [1, 1920, 1080, 3])
coords_ph = tf.placeholder(tf.float32, [None, 16, 9, 5])
obj_ph    = tf.placeholder(tf.float32, [None, 16, 9])
no_obj_ph = tf.placeholder(tf.float32, [None, 16, 9])

bn     = batch_norm(image_ph, 3, 'bn0')                   # 224 1920

pool1  = avg_pool(bn, 5)                                  #     1920
block1 = block(pool1, 3, 32, 3, 'block1')                 # 224 384

block2 = mobile_block(block1, 32, 64, 1, 'block2')        # 112 128
block3 = mobile_block(block2, 64, 128, 2, 'block3')       # 112 128

block4 = mobile_block(block3, 128, 128, 1, 'block4')      # 56  64
block5 = mobile_block(block4, 128, 256, 2, 'block5')      # 56  64

block6 = mobile_block(block5, 256, 256, 1, 'block6')      # 28  32
block7 = mobile_block(block6, 256, 512, 2, 'block7')      # 28  32

block8 = mobile_block(block7, 512, 512, 1, 'block8')      # 14  16
block9 = mobile_block(block8, 512, 512, 1, 'block9')      # 14  16
block10 = mobile_block(block9, 512, 512, 1, 'block10')    # 14  16
block11 = mobile_block(block10, 512, 512, 1, 'block11')   # 14  16
block12 = mobile_block(block11, 512, 512, 1, 'block12')   # 14  16

flat   = tf.reshape(block12, [1, 512*16*9])

mat1   = tf.Variable(init_matrix(size=(512*16*9, 4096), init='alexnet'), dtype=tf.float32, name='fc1')
bias1  = tf.Variable(np.zeros(shape=4096), dtype=tf.float32, name='fc1_bias')
fc1    = tf.matmul(flat, mat1) + bias1
relu1  = tf.nn.relu(fc1)

mat2   = tf.Variable(init_matrix(size=(4096, 16*9*10), init='alexnet'), dtype=tf.float32, name='fc2')
bias2  = tf.Variable(np.zeros(shape=16*9*10), dtype=tf.float32, name='fc2_bias')
fc2    = tf.matmul(relu1, mat2) + bias2

out    = tf.reshape(fc2, [1, 16, 9, 10])

###############################################################

loss = yolo_loss(out, coords_ph, obj_ph, no_obj_ph)
# train = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1.).minimize(loss)
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

        [p, l] = sess.run([out, loss], feed_dict={image_ph: image, coords_ph: coords, obj_ph: obj, no_obj_ph: no_obj})
        # print (np.shape(l))
        print (l)
        print (np.std(p), np.shape(p))

###############################################################



    
    
    
    
    
    


