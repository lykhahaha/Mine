import cv2

class MeanPreprocessor:
    def __init__(self, r_mean, g_mean, b_mean):
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean

    def preprocess(self, image):
        b, g, r = cv2.split(image.astype('float'))

        b -= self.b_mean
        g -= self.g_mean
        r -= self.r_mean

        return cv2.merge([b, g, r])