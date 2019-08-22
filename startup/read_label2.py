
import csv
import numpy as np

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

print (lookup['/home/brian/Documents/projects/object_detection/MOT17Det/train/MOT17-02/img1/000010.jpg'])
print (len(lookup.keys()))
