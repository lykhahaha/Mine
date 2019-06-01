import cv2

class SimplePreprocessor:
    def __init__(self, height, width, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.height = height
        self.width = width
        self.inter = inter
        
    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect
        # ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)