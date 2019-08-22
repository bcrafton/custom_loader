
import os

import csv
import numpy as np

import cv2
import matplotlib.pyplot as plt

#########################################

folders = [
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-02-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-02-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-02-SDP/',

'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-04-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-04-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-04-SDP/',

'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-05-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-05-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-05-SDP/',

'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-09-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-09-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-09-SDP/',

'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-10-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-10-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-10-SDP/',

'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-11-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-11-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-11-SDP/',

'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-11-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-11-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-11-SDP/',
]

#########################################

def get_images(path):

    images = []
    labels = {}

    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                for file in folder_files:
                    if 'jpg' in file:
                        images.append(os.path.join(folder_subdir, file))

    return images

#########################################

def get_labels(folders):
    lookup = {}
    for folder in folders:
        mat = np.loadtxt(open(folder + 'det/det.txt', "rb"), delimiter=",", skiprows=0)
        for label in mat:
            key = folder + ('img1/%06d.jpg' % (int(label[0])))
            if key in lookup:
                lookup[key].append(label)
            else:
                lookup[key] = [label]

    return lookup

#########################################

train_images = get_images('/home/brian/Documents/projects/object_detection/MOT17Det/train')
train_labels = get_labels(folders)

#########################################

filename = '/home/brian/Documents/projects/object_detection/MOT17/train/MOT17-02-DPM/img1/000100.jpg'
image  = cv2.imread(filename)
labels = train_labels[filename]

for label in labels:
    # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    l = int(label[2])
    t = int(label[3])
    w = int(label[4])
    h = int(label[5])
    c = int(label[6])
    image[l:(l+w), t:(t+h), :] = c

plt.imshow(image)
plt.show()






