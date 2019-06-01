import imutils
import cv2

class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # get the width and height image, initialize deltas to crop
        h, w = image.shape[:2]
        d_w, d_h = 0, 0

        # if width is smaller than height, resize along the width and update deltas
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            d_h = int((image.shape[0] - self.height)//2)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            d_w = int((image.shape[1] - self.width)//2)
        
        # crop the image based on deltas
        h, w = image.shape[:2]
        image = image[d_h:h-d_h, d_w:w-d_w]

        # finally, resize the image to ensure
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)