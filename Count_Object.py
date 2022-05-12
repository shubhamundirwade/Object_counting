# Importing Libraries

from itertools import count
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly

# Reading Image
image = cv2.imread("traffic.jpg")


# creating box over image, and label and counting objects in it 
box, label, count =  cv.detect_common_objects(image)

# Output 
output = draw_bbox(image, box, label, count)

plt.imshow(output)
plt.show()

print("Number of objects in this image: " + str(label.count('car')))
