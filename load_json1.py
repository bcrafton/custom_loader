
import json

filename = '/home/brian/Desktop/annotations/instances_train2014.json'

with open(filename) as json_file:
    data = json.load(json_file)

    images = list(data['images'])

    ids = {}
    for ii in range(len(images)):
        id = images[ii]['id'] # 82081
        if id not in ids.keys():
            ids[id] = id

    keys = list(ids.keys())
    print (len(keys))
    print (min(keys))
    print (max(keys))

    #######################################

    annotations = list(data['annotations'])

    ids = {}
    for ii in range(len(annotations)):
        id = annotations[ii]['image_id'] # 82081
        if id not in ids.keys():
            ids[id] = id


    keys = list(ids.keys())

    '''
    print (len(keys))
    print (min(keys))
    print (max(keys))
    '''
