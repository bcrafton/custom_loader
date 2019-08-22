
import csv
import numpy as np

import cv2
import matplotlib.pyplot as plt

folders = [
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-02-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-02-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-02-SDP/',

'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-04-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-04-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-04-SDP/',
]

lookup = {}
for folder in folders:
    mat = np.loadtxt(open(folder + 'det/det.txt', "rb"), delimiter=",", skiprows=0)
    for label in mat:
        key = folder + ('img1/%06d.jpg' % (int(label[0])))
        if key in lookup:
            lookup[key].append(label)
        else:
            lookup[key] = [label]

# print (np.shape(mat[:, 6]))
# print (np.shape(mat))
max_conf = np.max(mat[:, 6])

filename = '/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-02-DPM/img1/000100.jpg'
image  = cv2.imread(filename)
labels = lookup[filename]

for label in labels:
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    l = int(label[2])
    t = int(label[3])
    w = int(label[4])
    h = int(label[5])
    c = int(label[6])
    image[l:(l+w), t:(t+h), :] = c / max_conf * 255.

plt.imshow(image)
plt.show()

