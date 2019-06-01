import cv2
import imutils

class AspectAwarePreprocessor:
    def __init__(self, height, width, inter=cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.inter = inter
    
    def preprocess(self, image):
        h, w = image.shape[:2]

        if h > w:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            h, w = image.shape[:2]
            d_h = int((h - self.height)//2)
            image = image[d_h:h-d_h, :, :]
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            h, w = image.shape[:2]
            d_w = int((w - self.width)//2)
            image = image[:, d_w:w-d_w, :]
        
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)