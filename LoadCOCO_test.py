
from LoadCOCO import LoadCOCO
import matplotlib.pyplot as plt

loader = LoadCOCO()

while True:
    if not loader.empty():
        image, (coords, obj, no_obj) = loader.pop()
        plt.imshow(image)
        plt.show()
