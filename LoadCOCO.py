
import os

import csv
import numpy as np

import cv2

import queue
import threading

import json

#########################################

exxact = 0
icsrl2 = 1

if exxact:
    path = '/home/bcrafton3/Data_HDD/mscoco/'
elif icsrl2:
    path = '/usr/scratch/bcrafton/mscoco/'
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

# we need a class remap table because COCO has 80 classes ... but the ids are not ordered.
# max(class_id) = 90 ... even though there are only 80 classes.
def get_cat_table(json_filename):
    table = {}

    json_file = open(json_filename)
    data = json.load(json_file)
    annotations = list(data['annotations'])
    json_file.close()

    cats = {}
    for annotation in annotations:
        id = annotation['category_id']
        if id not in cats.keys():
            cats[id] = id

    sorted_cats = sorted(cats.keys())
    for ii in range(len(sorted_cats)):
        table[sorted_cats[ii]] = ii

    return table

#########################################

def get_det_table(json_filename):
    table = {}

    json_file = open(json_filename)
    data = json.load(json_file)
    annotations = list(data['annotations'])
    json_file.close()

    for annotation in annotations:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        cat_id = annotation['category_id']
        image_filename = path + 'train_images/COCO_train2014_%012d.jpg' % (int(image_id))

        new_det = (bbox, cat_id)

        if image_filename in table.keys():
            table[image_filename].append(new_det)
        else:
            table[image_filename] = [new_det]

    return table

#########################################

def preprocess(det_size, filename, det_table, cat_table):
    # image
    image = cv2.imread(filename)
    shape = np.shape(image)
    (w, h, _) = shape
    image = cv2.resize(image, (448, 448))
    # image = np.reshape(image, [1, 448, 448, 3])
    scale_w = 448 / w
    scale_h = 448 / h

    # dets
    dets = det_table[filename]
    coords  = np.zeros(shape=[det_size, 7, 7, 5])
    obj     = np.zeros(shape=[det_size, 7, 7])
    no_obj  = np.ones(shape=[det_size, 7, 7])
    cats    = np.zeros(shape=[det_size, 7, 7])

    ndets = min(len(dets), det_size)

    for ii in range(ndets):
        det = dets[ii]
        [y, x, h, w], cat_id = det

        cat = cat_table[cat_id]

        x = x * scale_w
        y = y * scale_h
        w = w * scale_w
        h = h * scale_h

        if not (x <= 448.1 and y <= 448.1 and w <= 448.1 and h <= 448.1):
            print (x, y, w, h, shape, scale_w, scale_h)
            assert(x <= 448.1 and y <= 448.1 and w <= 448.1 and h <= 448.1)

        xc = int(x) // 64
        yc = int(y) // 64

        # should put more asserts in here...
        # make sure we are between 0 and 64 before / 64.

        x = (x - xc * 64.) / 64. # might want to clip this to zero
        y = (y - yc * 64.) / 64. # might want to clip this to zero
        w = w / 448.
        h = h / 448.

        coords[ii, xc, yc, :] = np.array([x, y, w, h, 1.])
        obj[ii, xc, yc] = 1.
        no_obj[ii, xc, yc] = 0.
        cats[ii, xc, yc] = cat

    return image, (coords, obj, no_obj, cats)

#########################################

def fill_queue(batch_size, det_size, images, det_table, cat_table, q):
    ii = 0
    last = len(images) - 1

    while(True):
        if not q.full():
            image_batch  = []
            coords_batch = []
            obj_batch    = []
            no_obj_batch = []
            cats_batch   = []

            while (len(image_batch) < batch_size):
                filename = images[ii]

                ii = (ii + 1) if ((ii + 1) < last) else 0

                if filename in det_table.keys():
                    image, (coords, obj, no_obj, cats) = preprocess(det_size, filename, det_table, cat_table)
                else:
                    continue

                image_batch.append(image)
                coords_batch.append(coords)
                obj_batch.append(obj)
                no_obj_batch.append(no_obj)
                cats_batch.append(cats)

            image_batch = np.stack(image_batch)
            coords_batch = np.stack(coords_batch)
            obj_batch = np.stack(obj_batch)
            no_obj_batch = np.stack(no_obj_batch)
            cats_batch = np.stack(cats_batch)

            image = image_batch
            det = (coords_batch, obj_batch, no_obj_batch, cats_batch)

            q.put((image, det))

#########################################

class LoadCOCO:

    def __init__(self, batch_size, det_size):
        self.batch_size = batch_size
        self.det_size = det_size

        self.cat_table = get_cat_table(path + 'train_labels/instances_train2014.json')
        self.train_images = sorted(get_images(path + 'train_images'))
        self.train_det_table = get_det_table(path + 'train_labels/instances_train2014.json')

        '''
        maxval = 0
        for key in self.train_det_table.keys():
            val = self.train_det_table[key]
            if len(val) > maxval:
                maxval = len(val)
                maxkey = key
        '''

        self.q = queue.Queue(maxsize=128)
        thread = threading.Thread(target=fill_queue, args=(self.batch_size, self.det_size, self.train_images, self.train_det_table, self.cat_table, self.q))
        thread.start()

    def pop(self):
        return self.q.get()

    def empty(self):
        return self.q.empty()

    def full(self):
        return self.q.full()

###################################################################

















