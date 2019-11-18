
from LoadCOCO import LoadCOCO
import matplotlib.pyplot as plt
import numpy as np

loader = LoadCOCO()

batch = []
while True:
    if not loader.empty():
        image, label = loader.pop()
        batch.append((image, label))

        if len(batch) == 4:
            images = []; 
            coords = []; objs = []; no_objs = []; cats = []
            for (image, label) in batch:
                images.append(image)
                (coord, obj, no_obj, cat) = label
                coords.append(coord); objs.append(obj); no_objs.append(no_obj); cats.append(cat)

            images = np.concatenate(images, axis=0)
            coords = np.concatenate(coords, axis=0)
            objs = np.concatenate(objs, axis=0)
            no_objs = np.concatenate(no_objs, axis=0)
            cats = np.concatenate(cats, axis=0)
            print (np.shape(images))
            print (np.shape(objs))
            assert (False)
