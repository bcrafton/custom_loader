
import os

import csv
import numpy as np

import cv2
import matplotlib.pyplot as plt

#########################################

train_folders = [
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

test_folders = [
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-01-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-01-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-01-SDP/',

'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-03-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-03-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-03-SDP/',

'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-06-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-06-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-06-SDP/',

'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-07-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-07-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-07-SDP/',

'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-08-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-08-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-08-SDP/',

'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-12-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-12-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-12-SDP/',

'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-14-DPM/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-14-FRCNN/',
'/home/brian/Documents/projects/object_detection/MOT17/test/MOT17-14-SDP/',
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

def fill_queue(d, q):
    ii = 0
    last = len(d) - 1

    while(True):
        if q.full() == False:
            filename = d[ii]
            x = cv2.imread(filename)
            q.put(x)
            ii = (ii + 1) if (ii < last) else 0
            print (ii)

#########################################

class LoadMOT:

    def __init__(self):
        self.train_images = get_images('/home/brian/Documents/projects/object_detection/MOT17Det/train')
        self.train_labels = get_labels(train_folders)

        self.test_images = get_images('/home/brian/Documents/projects/object_detection/MOT17Det/test')
        self.test_labels = get_labels(test_folders)

        self.q = queue.Queue(maxsize=128)
        thread = threading.Thread(target=fill_queue, args=(self.train_images, self.q))
        thread.start()

    def pop(self):
        return self.q.get()

###################################################################

















