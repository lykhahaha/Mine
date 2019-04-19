from keras.models import Model
from keras.layers import Flatten, Dense, Dropout

class FCHeadNet:
    def build(base_model, classes, D):
        # initialize the head model that will be placed on top of base model
        head_model = base_model.output
        head_model = Flatten()(head_model)
        head_model = Dense(D, activation='relu')(head_model)
        head_model = Dropout(0.5)(head_model)

        # add softmax layer
        head_model = Dense(classes, activation='softmax')(head_model)

        return head_model