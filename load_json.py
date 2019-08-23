
import json

filename = '/home/brian/Desktop/annotations/instances_train2014.json'

with open(filename) as json_file:
    data = json.load(json_file)

    keys = list(data.keys())
    print (keys)

    annotations = list(data['annotations'])
    print (len(annotations))
    print (annotations[0])

    images = list(data['images'])
    print (len(images))
    print (images[0])
