
import json

filename = '/home/brian/Desktop/annotations/instances_train2014.json'

with open(filename) as json_file:
    data = json.load(json_file)

    keys = list(data.keys())
    print (keys)

    images = list(data['images'])
    print (len(images))
    print (images[0])

    annotations = list(data['annotations'])
    print (len(annotations))

    for ii in range(10):
        print (annotations[ii])
