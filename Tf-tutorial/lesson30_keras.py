import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, models, metrics

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.
    y = tf.cast(y, dtype=tf.int32)

    return x, y

BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 30

(train_x, train_y), (val_x, val_y) = datasets.mnist.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.map(preprocess).shuffle(60000).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.map(preprocess).batch(BATCH_SIZE)

model = models.Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(21, activation=tf.nn.relu),
    layers.Dense(10)
])

class MyDense(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.kernel = self.add_variable('w', [input_dim, output_dim])
        self.bias = self.add_variable('b', [output_dim])
    
    def call(self, inputs, training=None):
        out = inputs@self.kernel + self.bias
        return out

class MyModel(models.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = MyDense(28*28, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)
    
    def call(self, inputs, training=None):
        x = tf.nn.relu(self.fc1(inputs))
        x = tf.nn.relu(self.fc2(x))
        x = tf.nn.relu(self.fc3(x))
        x = tf.nn.relu(self.fc4(x))
        x = self.fc5(x)
        return x
        

model.build(input_shape=(None, 28*28))
model.summary()
# Use compile
model.compile(optimizer=optimizers.Adam(lr=LR),
            loss=tf.losses.categorical_crossentropy(from_logits=True),
            metrics=['accuracy'])

model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, validation_freq=2)
model.evaluate(val_dataset)