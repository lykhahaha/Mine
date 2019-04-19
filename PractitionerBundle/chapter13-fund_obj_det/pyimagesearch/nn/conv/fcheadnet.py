from keras.layers import Dropout, Flatten, Dense

class FCHeadNet:
    @staticmethod
    def build(base_model, classes, D):
        # initialize head model that will be placed on top of the base, then add FC layer
        head_model = base_model.output # Tensor("block5_pool/MaxPool:0", shape=(?, ?, ?, 512), dtype=float32)
        head_model = Flatten(name='flatten')(head_model)
        head_model = Dense(D, activation='relu')(head_model)
        head_model = Dropout(0.5)(head_model)

        # add softmax layer
        head_model = Dense(classes, activation='softmax')(head_model)

        return head_model