import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

import numpy as np
import tensorflow as tf
import cv2
from dataset import Dataset

tf.logging.set_verbosity(tf.logging.INFO)

class LeNet:
    def __init__(self, weights=None, sess=None, log=True):
        self.X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='X')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.log = log
        self.sess = sess

        self.conv_layers()
        self.fc_layer()

        self.probs = tf.nn.softmax(self.logits, name='softmax')

    def conv_layers(self):
        self.parameters = []
        images = self.X

        with tf.name_scope('conv1') as scope:
            kernel_shape = (3, 3, 1, 32)
            kernel = tf.Variable(np.random.normal(0, np.sqrt(2.0/np.sum(kernel_shape)), kernel_shape), name='weights', dtype=tf.float32)
            conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(np.zeros(shape=[32]), dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        with tf.name_scope('conv2') as scope:
            kernel_shape = (3, 3, 32, 64)
            kernel = tf.Variable(np.random.normal(0, np.sqrt(2.0/np.sum(kernel_shape)), kernel_shape), name='weights', dtype=tf.float32)
            conv = tf.nn.conv2d(self.pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(np.zeros(shape=[64]), dtype=tf.float32, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    def fc_layer(self):
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool2.get_shape()[1:]))
            fc1w_shape = (shape, 128)
            fc1w = tf.Variable(np.random.normal(0, np.sqrt(2.0/np.sum(fc1w_shape)), fc1w_shape), name='weights', dtype=tf.float32)
            fc1b = tf.Variable(np.ones(shape=[128]), dtype=tf.float32, name='biases')
            # print(self.pool2.get_shape())

            pool2_flat = tf.reshape(self.pool2, [-1, shape])

            # print(pool2_flat.shape)
            fc1l = tf.nn.bias_add(tf.matmul(pool2_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.dropout1 = tf.nn.dropout(self.fc1, keep_prob=self.keep_prob, name='dropout1')
            self.parameters += [fc1w, fc1b]

        with tf.name_scope('fc2') as scope:
            fc2w_shape = (128, 10)
            fc2w = tf.Variable(np.random.normal(0, np.sqrt(2.0/np.sum(fc2w_shape)), fc2w_shape), name='weights', dtype=tf.float32)
            fc2b = tf.Variable(np.ones(shape=[10]), dtype=tf.float32, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.dropout1, fc2w), fc2b)
            self.logits = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

    def load_weights(weights, sess):
        None

    def train(self, learning_rate, training_epochs, batch_size, keep_prob):
        self.dataset = Dataset()
        self.Y = tf.placeholder(tf.float32, shape=[None, 10], name='Y')
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        if self.log:
            tf.summary.scalar('cost', self.cost)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('./log_train/exp1', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        print('Training...')
        weights = []

        for epoch in range(training_epochs):
            avg_cost = 0

            total_batch = int(self.dataset.get_train_set_size()/batch_size)
            for i in range(total_batch + 1):
                batch_xs, batch_ys = self.dataset.next_batch_train(batch_size)
                feed_dict = {
                    self.X: batch_xs.reshape([batch_xs.shape[0], 28, 28, 1]),
                    self.Y: batch_ys,
                    self.keep_prob: keep_prob
                }
                weights, summary, c, _ = self.sess.run([self.parameters, self.merged, self.cost, self.optimizer], feed_dict=feed_dict)
                avg_cost += c / total_batch

            if self.log:
                self.train_writer.add_summary(summary, epoch + 1)
            print('Epoch:', '%02d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        print('Training finished!')

        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "/tmp/mnist_brain.ckpt")
        print("Trainned model is saved in file: %s" % save_path)

    def evaluate(self, batch_size, keep_prob):

        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        N = self.dataset.get_test_set_size()
        print('test.size', N);
        correct_sample = 0
        for i in range(0, N, batch_size):
            batch_xs, batch_ys = self.dataset.next_batch_test(batch_size)

            N_batch = batch_xs.shape[0]

            feed_dict = {
                self.X: batch_xs.reshape([N_batch, 28, 28, 1]),
                self.Y: batch_ys,
                self.keep_prob: keep_prob
            }

            correct = self.sess.run(self.accuracy, feed_dict=feed_dict)
            correct_sample += correct * N_batch

        test_accuracy = correct_sample / N

        print("\nAccuracy Evaluates")
        print("-" * 30)
        print('Test Accuracy:', test_accuracy)

sess = tf.Session()
lenet = LeNet(sess=sess, weights=None)

lenet.train(learning_rate=0.001, training_epochs=40, batch_size=1000, keep_prob=0.7)
lenet.evaluate(batch_size=1000, keep_prob=0.7)