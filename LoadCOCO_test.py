
from LoadCOCO import LoadCOCO
import matplotlib.pyplot as plt
import numpy as np

loader = LoadCOCO()

while True:
    if not loader.empty():
        image, (coords, objs, no_objs) = loader.pop()

        print (np.shape(coords), np.shape(objs), np.shape(no_objs))

        for ii in range(len(coords)):

            coord = coords[ii]
            obj = objs[ii]

            flag = False
            for jj in range(7):
                for kk in range(7):
                    if obj[jj][kk]:
                        assert(not flag)
                        [x, y, w, h, _] = coord[jj][kk]
                        flag = True
            assert(flag)

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            image = image / np.max(image)
            top = image
            bottom = np.copy(image)
            bottom[x:(x+w), y:(y+h), :] = 1.0
            concat = np.concatenate((top, bottom), axis=1)
            plt.imsave('%d.jpg' % (ii), concat)

