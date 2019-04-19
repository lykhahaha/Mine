import imutils
import cv2

class AspectAwarePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width # 256
        self.height = height # 256
        self.inter = inter
    
    def preprocess(self, image):
        # grab the dimension of image, then initialize deltas to use when cropping
        h, w = image.shape[:2] # h = 500, w = 557
        dW, dH = 0, 0
        # if width is smaller than height, resize along width and then update deltas to crop height to desired dimension
        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height)/2.0) # 310-256/2=27
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter) # (256, 285)
            dW = int((image.shape[1] - self.width)/2.0) # (285-256)/2=13
        # now that our images have been resized, we need to re-grab width and height, followed by cropping
        h, w = image.shape[:2]
        image = image[dH:h-dH, dW:w-dW] # (256, 257)
        # finally, resize the image to the provided spatial dimensions to ensure our output image is always a fixed size
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)