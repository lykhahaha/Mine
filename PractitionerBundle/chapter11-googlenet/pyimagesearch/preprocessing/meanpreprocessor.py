import cv2

class MeanPreprocessor:
    def __init__(self, r_mean, g_mean, b_mean):
        # store the Red, Green, Blue channel averages across training set
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean

    def preprocess(self, image):
        # split image into its respective Red, Green , Blue channels
        # https://forums.fast.ai/t/images-normalization/4058/7
        B, G, R = cv2.split(image.astype('float32'))

        # subtract the means for each channel
        R -= self.r_mean
        G -= self.g_mean
        B -= self.b_mean

        # merge the channels back together and return image
        return cv2.merge([B, G, R])