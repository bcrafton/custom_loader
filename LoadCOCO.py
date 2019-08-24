
import os

import csv
import numpy as np

import cv2

import queue
import threading

import json

#########################################

exxact = 1
local = 0

if exxact:
    path = '/home/bcrafton3/Data_HDD/mscoco/'
elif local:
    assert(False)
else:
    assert(False)

#########################################

def get_images(path):

    images = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if 'jpg' in file:
                full_path = os.path.join(subdir, file)
                if (full_path not in images):
                    images.append(full_path)

    return images

#########################################

def get_labels_table(json_filename):
    table = {}

    json_file = open(json_filename)
    data = json.load(json_file)
    annotations = list(data['annotations'])
    json_file.close()

    for annotation in annotations:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        image_filename = path + 'train_images/COCO_train2014_%012d.jpg' % (int(image_id))

        if image_filename in table.keys():
            table[image_filename].append(bbox)
        else:
            table[image_filename] = [bbox]

    return table

#########################################

def preprocess(filename, table):
    # image
    image = cv2.imread(filename)
    (w, h, _) = np.shape(image)
    image = cv2.resize(image, (448, 448))
    scale_w = 448 / w
    scale_h = 448 / h

    # labels
    labels = table[filename]
    nlabels = len(labels)
    coords  = np.zeros(shape=[nlabels, 16, 9, 5])
    obj     = np.zeros(shape=[nlabels, 16, 9])
    no_obj  = np.ones(shape=[nlabels, 16, 9])

    for ii in range(nlabels):
        label = labels[ii]
        [x, y, w, h] = label

        x = x * scale_w
        y = y * scale_h
        w = w * scale_w
        h = h * scale_h

        xc = int(x) // 64
        yc = int(y) // 64

        coords[ii, xc, yc, :] = np.array([x, y, w, h, 1.])
        obj[ii, xc, yc] = 1.
        no_obj[ii, xc, yc] = 0.

    return image, (coords, obj, no_obj)

#########################################

def fill_queue(images, table, q):
    ii = 0
    last = len(images) - 1

    while(True):
        if not q.full():
            filename = images[ii]
            ii = (ii + 1) if (ii < last) else 0

            if filename in table.keys():
                image, label = preprocess(filename, table)
            else:
                print ('no label: %s' % (filename))
                continue

            q.put((image, label))

#########################################

class LoadCOCO:

    def __init__(self):
        self.train_images = sorted(get_images(path + 'train_images'))
        # self.test_images = sorted(get_images(path + 'test'))

        self.train_labels_table = get_labels_table(path + 'train_labels/instances_train2014.json')
        # self.test_labels_table = get_labels_table(test_folders)

        # print (self.train_labels_table.keys())

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

















