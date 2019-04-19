from pyimagesearch.utils.simple_obj_det import non_max_suppression
import numpy as np
import cv2

# image = cv2.imread('bksomels.jpg')
# orig = image.copy()

bounding_box = np.array([
    (114, 60, 178, 124),
    (120, 60, 184, 124),
    (114, 66, 178, 130)
])

pick = non_max_suppression(bounding_box, 0.3)
print(pick)