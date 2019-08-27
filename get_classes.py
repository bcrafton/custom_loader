
import json

filename = '/home/brian/Desktop/annotations/instances_train2014.json'

with open(filename) as json_file:
    data = json.load(json_file)

    images = list(data['images'])
    labels = list(data['annotations'])

    classes = {}
    for label in labels:
        id = label['category_id']
        if id not in classes.keys():
            classes[id] = id

    classes = sorted(classes.keys())
    print (len(classes), classes)


