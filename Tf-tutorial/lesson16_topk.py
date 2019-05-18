import tensorflow as tf

def accuracy(output, target, top_k=(1,)):
    """
    output : [10, 6]
    target: [10]
    """
    max_k = max(top_k)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, max_k).indices # [10, 6]
    pred = tf.transpose(pred, perm=[1, 0]) # [6, 10] to compare top_1, top_2,... between pred and target

    target = tf.broadcast_to(target, pred.shape)
    correct = tf.equal(pred, target)

    res = []
    for k in top_k:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]), dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k * (100 / batch_size))
        res.append(acc)

    return res

output = tf.random.normal([10, 6])
output = tf.math.softmax(output, axis=1)
target = tf.random.uniform([10], maxval=6, dtype=tf.int32)
acc = accuracy(output, target, top_k=(1,2,3,4,5,6))