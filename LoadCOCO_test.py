
from LoadCOCO import LoadCOCO
import matplotlib.pyplot as plt
import numpy as np

loader = LoadCOCO()

counter = 0
while True:
    if not loader.empty():
        image, (coords, objs, no_objs) = loader.pop()
        image = np.squeeze(image)

        for ii in range(len(coords)):
            coord = coords[ii]
            obj = objs[ii]

            [xc, yc] = np.squeeze(np.where(obj > 0))
            [x, y, w, h, _] = coord[xc][yc]

            x = int(x * 64. + xc * 64.)
            y = int(y * 64. + yc * 64.)
            w = int(w * 448.)
            h = int(h * 448.)

            image = image / np.max(image)
            top = image
            bottom = np.copy(image)
            bottom[x:(x+w), y:(y+h), :] = 0.
            concat = np.concatenate((top, bottom), axis=1)

            plt.imsave('%d.jpg' % (counter), concat)
            break

        print (counter)
        counter = counter + 1
