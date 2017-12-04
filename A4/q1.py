import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_weights_relu(shape, name=""):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.071), name)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden, act):
    with tf.name_scope('Layer_1'):
        with tf.name_scope('Activation'):
            l1a = act(tf.nn.conv2d(X, w,                       # l1a shape=(?, 32, 32, 32)
                                strides=[1, 1, 1, 1], padding='SAME'))
            variable_summaries(l1a)

        with tf.name_scope('Pooling'):
            l1 = tf.nn.max_pool(l1a, ksize=[1, 4, 4, 1],              # l1 shape=(?, 16, 16, 32)
                                strides=[1, 2, 2, 1], padding='SAME')
            variable_summaries(l1)

        with tf.name_scope('Dropout'):
            tf.summary.scalar('dropout_keep_probabiliy', p_keep_conv)

            l1 = tf.nn.dropout(l1, p_keep_conv)
            variable_summaries(l1)

    with tf.name_scope('Layer_2'):
        with tf.name_scope('Activation'):
            l2a = act(tf.nn.conv2d(l1, w2,                       # l2a shape=(?, 16, 16, 64)
                                strides=[1, 1, 1, 1], padding='SAME'))
            variable_summaries(l2a)

        with tf.name_scope('Pooling'):
            l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 8, 8, 64)
                                strides=[1, 2, 2, 1], padding='SAME')
            variable_summaries(l2)

        with tf.name_scope('Dropout'):
            tf.summary.scalar('dropout_keep_probabiliy', p_keep_conv)

            l2 = tf.nn.dropout(l2, p_keep_conv)
            variable_summaries(l2)

    l3 = tf.reshape(l2, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 16x16x32)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    with tf.name_scope('FinalActivations'):

        l4 = tf.matmul(l3, w_fc)
        tf.summary.histogram('pre-activ', l4)

        l4 = act(l4)
        tf.summary.histogram('activ', l4)
   
        l4 = tf.nn.dropout(l4, p_keep_hidden)

        pyx = tf.matmul(l4, w_o)
        tf.summary.histogram('out-layer', pyx)

    return pyx

(trX, trY_), (teX, teY_) = cifar10.load_data()
trX = trX.reshape(-1, 32, 32, 3)  # 32x32x3 input img
trX = trX.astype('float32')
teX = teX.reshape(-1, 32, 32, 3)  # 32x32x3 input img
teX = teX.astype('float32')

#trY = np.zeros([trY_.shape[0], 10])
#for i in range(trY.shape[0]):
#    trY[i][trY_[i][0]] = 1

#teY = np.zeros([teY_.shape[0], 10])
#for i in range(teY.shape[0]):
#    teY[i][teY_[i][0]] = 1

num_classes = len(np.unique(trY))
trY = np_utils.to_categorical(trY_, num_classes)
teY = np_utils.to_categorical(teY_, num_classes)

X = tf.placeholder("float", [None, 32, 32, 3])
Y = tf.placeholder("float", [None, 10])

w = init_weights_relu([3, 3, 3, 32], "Layer_1_Weights")       # 3x3x3 conv, 32 outputs
w2 = init_weights_relu([3, 3, 32, 64], "Layer_2_Weights")      # 3x3x32 conv, 64 out
w_fc = init_weights_relu([64 * 8 * 8, 625], "FC_Weights") # FC 64 * 8 * 8 inputs, 625 outputs
w_o = init_weights_relu([625, 10], "Output_Weights")         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden, tf.nn.relu)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(15):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            with tf.name_scope('train-batch'):
                sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                              p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./train', sess.graph)

    test_writer = tf.summary.FileWriter('./test', sess.graph)