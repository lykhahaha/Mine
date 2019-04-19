from config import age_gender_config as config
import cv2
from keras.models import load_model
import numpy as np
from scipy import misc

class SuperResolutionPreprocesor:
    def __init__(self, height, width, inter=cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.inter = inter
        self.model = load_model(config.MODEL_PATH)

    def preprocess(self, image):
        h, w = image.shape[:2]
        print(self.height, h, self.width, w)
        if h < w:
            SCALE = self.height//h
        else:
            SCALE = self.width//w
        
        INPUT_DIM = 33

        # label size should be output spatial dimensions of SRCNN while padding ensures we properly crop label ROI
        LABEL_SIZE = 21
        PAD = int((INPUT_DIM - LABEL_SIZE) / 2.)
        # print(SCALE)
        scaled = misc.imresize(image, 2., interp='bicubic')
        output = np.zeros(scaled.shape)
        h, w = output.shape[:2] 

        # slide a window from left-to-right and top-to-bottom
        for y in range(0, h - INPUT_DIM + 1, LABEL_SIZE):
            for x in range(0, w - INPUT_DIM + 1, LABEL_SIZE):
                # crop ROI from our scaled image
                crop = scaled[y:y + INPUT_DIM, x:x + INPUT_DIM]

                # predict on crop and store it
                P = self.model.predict(np.expand_dims(crop, axis=0))
                P = P.reshape((LABEL_SIZE, LABEL_SIZE, 3))
                output[y + PAD: y + PAD + LABEL_SIZE, x + PAD: x + PAD + LABEL_SIZE] = P

        # remove any of black borders in output image caused by padding, then clip any values that fall outside range [0, 255]        
        output = output[PAD:h - (h%INPUT_DIM + PAD), PAD:w - (w%INPUT_DIM + PAD)]
        output = np.clip(output, 0, 255).astype('uint8')

        cv2.imwrite('scaled.jpg', image)

        return cv2.resize(output, (self.width, self.height), interpolation=self.inter)