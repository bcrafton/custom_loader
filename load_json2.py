
import json

filename = '/home/brian/Desktop/annotations/instances_train2014.json'

with open(filename) as json_file:
    data = json.load(json_file)

    images = list(data['images'])
    labels = list(data['annotations'])

    label = labels[0]
    
    for image in images:
        if image['id'] == label['image_id']:
            break

    print (image) 
    print (label)


