
import csv
import numpy as np

import cv2
import matplotlib.pyplot as plt

folders = [
'/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-02/',
'/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-04/',
'/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-05/',
'/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-09/',
'/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-10/',
'/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-11/',
'/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-13/',
]

lookup = {}
for folder in folders:
    mat = np.loadtxt(open(folder + 'gt/gt.txt', "rb"), delimiter=",", skiprows=0)
    for label in mat:
        key = folder + ('img1/%06d.jpg' % (int(label[0])))
        if key in lookup:
            lookup[key].append(label)
        else:
            lookup[key] = [label]

filename = '/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-02/img1/000052.jpg'
image  = cv2.imread(filename)
labels = lookup[filename]

for label in labels:
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    l = int(label[2])
    t = int(label[3])
    w = int(label[4])
    h = int(label[5])
    c = int(label[6])
    image[l:(l+w), t:(t+h), :] = c * 255.

plt.imshow(image)
plt.show()

