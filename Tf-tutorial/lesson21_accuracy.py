import tensorflow as tf

def accuracy(w_1, b_1, w_2, b_2, w_3, b_3, dataset):
    total_correct = 0
    total = 0
    for step, (x, y) in enumerate(dataset):
        h_1 = x@w_1 +b_1
        h_1 = tf.nn.relu(h_1)
        # [BATCH_SIZE, 256]@[256, 128]+[128] = [BATCH_SIZE, 128]
        h_2 = h_1@w_2 + b_2
        h_2 = tf.nn.relu(h_2)
        # [BATCH_SIZE, 128]@[128, 10]+[10] = [BATCH_SIZE, 10]
        out = h_2@w_3 + b_3

        pred = tf.argmax(out, axis=1)

        y = tf.argmax(y, axis=1)

        correct = tf.equal(pred, y)

        total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
        total += x.shape[0]
    
    return total_correct/total