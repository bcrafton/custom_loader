
import os

import csv
import numpy as np

import cv2

import queue
import threading

#########################################

exxact = 1
local = 0

if exxact:
    path = '/home/bcrafton3/MOT17/'
elif local:
    path = '/home/brian/Documents/projects/object_detection/MOT17/'
else:
    assert(False)

train_folders = [
path + 'train/MOT17-02-DPM/',
path + 'train/MOT17-02-FRCNN/',
path + 'train/MOT17-02-SDP/',

path + 'train/MOT17-04-DPM/',
path + 'train/MOT17-04-FRCNN/',
path + 'train/MOT17-04-SDP/',

path + 'train/MOT17-05-DPM/',
path + 'train/MOT17-05-FRCNN/',
path + 'train/MOT17-05-SDP/',

path + 'train/MOT17-09-DPM/',
path + 'train/MOT17-09-FRCNN/',
path + 'train/MOT17-09-SDP/',

path + 'train/MOT17-10-DPM/',
path + 'train/MOT17-10-FRCNN/',
path + 'train/MOT17-10-SDP/',

path + 'train/MOT17-11-DPM/',
path + 'train/MOT17-11-FRCNN/',
path + 'train/MOT17-11-SDP/',

path + 'train/MOT17-11-DPM/',
path + 'train/MOT17-11-FRCNN/',
path + 'train/MOT17-11-SDP/',
]

test_folders = [
path + 'test/MOT17-01-DPM/',
path + 'test/MOT17-01-FRCNN/',
path + 'test/MOT17-01-SDP/',

path + 'test/MOT17-03-DPM/',
path + 'test/MOT17-03-FRCNN/',
path + 'test/MOT17-03-SDP/',

path + 'test/MOT17-06-DPM/',
path + 'test/MOT17-06-FRCNN/',
path + 'test/MOT17-06-SDP/',

path + 'test/MOT17-07-DPM/',
path + 'test/MOT17-07-FRCNN/',
path + 'test/MOT17-07-SDP/',

path + 'test/MOT17-08-DPM/',
path + 'test/MOT17-08-FRCNN/',
path + 'test/MOT17-08-SDP/',

path + 'test/MOT17-12-DPM/',
path + 'test/MOT17-12-FRCNN/',
path + 'test/MOT17-12-SDP/',

path + 'test/MOT17-14-DPM/',
path + 'test/MOT17-14-FRCNN/',
path + 'test/MOT17-14-SDP/',
]

#########################################

def get_images(path):

    images = []
    for subdir, dirs, files in os.walk(path):
        for folder in dirs:
            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                for file in folder_files:
                    if 'jpg' in file:
                        images.append(os.path.join(folder_subdir, file))

    return images

#########################################

def get_labels_table(folders):
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

def get_boxes(labels):
    nlabels = len(labels)
    obj     = np.zeros(shape=[nlabels, 16, 9])
    no_obj  = np.zeros(shape=[nlabels, 16, 9])
    coords  = np.zeros(shape=[nlabels, 16, 9, 5])

    for ii in range(nlabels):
        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        
        label = labels[ii]

        l = max(label[2], 0.)
        t = max(label[3], 0.)
        w = max(label[4], 0.)
        h = max(label[5], 0.)
        c = max(label[6], 0.)

        x = int(l) // 120
        y = int(t) // 120

        coords[ii, x, y, :] = np.array([l, t, h, w, c])
        obj[ii, x, y] = 1
        no_obj[ii] = np.ones(shape=[16, 9]) - obj[ii]

    return (coords, obj, no_obj)

#########################################

def fill_queue(images, labels_table, q):
    ii = 0
    last = len(images) - 1

    while(True):
        if not q.full():
            filename = images[ii]
            x = cv2.imread(filename)
            y = get_boxes(labels_table[filename])
            q.put((x, y))
            ii = (ii + 1) if (ii < last) else 0

#########################################

class LoadMOT:

    def __init__(self):
        self.train_images = get_images(path + 'train')
        self.train_labels_table = get_labels_table(train_folders)

        self.test_images = get_images(path + 'test')
        self.test_labels_table = get_labels_table(test_folders)

        self.q = queue.Queue(maxsize=128)
        thread = threading.Thread(target=fill_queue, args=(self.train_images, self.train_labels_table, self.q))
        thread.start()

    def pop(self):
        return self.q.get()

    def empty(self):
        return self.q.empty()

    def full(self):
        return self.q.full()

###################################################################

















