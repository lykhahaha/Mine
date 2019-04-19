import numpy as np

class CropPreprocessor:
    def __init__(self, width, height, horiz=True):
        self.width = width
        self.height = height
        self.horiz = horiz

    def preprocess(self, image):
        h, w = image.shape[:2]

        coords = [
            (0, 0 , self.width, self.height),
            (w-self.width, 0, w, self.height),
            (0, h-self.height, self.width, h),
            (w-self.width, h-self.height, w, h)
        ]

        dW = int((w - self.width)//2.)
        dH = int((h - self.height)//2.)
        coords.append([dW, dH, w - dW, h - dH])

        crops = []
        for start_x, start_y, end_x, end_y in coords:
            crop = image[start_y:end_y, start_x:end_x]
            crops.append(crop)

        if self.horiz:
            mirrors = [cv2.flip(c, 1) for c in crops]
            crops.extend(mirrors)

        return np.array(crops)