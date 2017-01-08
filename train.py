import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import dataset
import os

from numpy import genfromtxt

path_model = "model.ckpt"


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 1  # -inf inf NaN
    return c


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    nan_indices = (mu == 0)
    result = div0((dataset - mu), sigma)
    # remove useless columns
    result = result[:, ~nan_indices]
    return result


def append_bias_reshape(features, labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    # insert ones into every row of features, so the array would be [1, x0, x1, x2],
    # and the reshaped shape would be [n_training_samples x (n_dim + 1)]
    # >>> np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
    # >>> array([[1, 2, 3, 0, 0, 4, 5, 6]])
    f = np.reshape(np.c_[np.ones(n_training_samples), features], [n_training_samples, n_dim + 1])
    l = np.reshape(labels, [n_training_samples, 1])
    return f, l


def extract_weights(sess, W):
    with sess.as_default():
        weights = W.eval()
        return weights


def prepare_data():
    # load features and labels
    features, labels = dataset.extract_features()
    print features.shape
    print labels.shape
    # normalize features
    normalized_features = feature_normalize(features)
    # append bias to features and labels
    f, l = append_bias_reshape(normalized_features, labels)
    # number of dimensions (14)
    print f.shape
    print l.shape

    # randomly chose 80% of data as train, 20% as test
    rnd_indices = np.random.rand(len(f)) < 0.80

    train_x = f[rnd_indices]
    train_y = l[rnd_indices]
    test_x = f[~rnd_indices]
    test_y = l[~rnd_indices]
    return train_x, test_x, train_y, test_y


BAD_THRESHOLD = 60


def measure_accuracy(test_y, pred_y):
    bad_hit_indices = np.logical_and(test_y <= BAD_THRESHOLD, pred_y <= BAD_THRESHOLD)
    innocent_indices = np.logical_and(test_y > BAD_THRESHOLD, pred_y <= BAD_THRESHOLD)
    appropriate_indices = np.logical_and(test_y - pred_y <= 10, test_y - pred_y >= -10)

    bad_hit_accuracy = float(len(test_y[bad_hit_indices])) / float(len(test_y))
    innocent_rate = float(len(test_y[innocent_indices])) / float(len(test_y))
    appropriate_accuracy = float(len(test_y[appropriate_indices])) / float(len(test_y))

    print "bad hit accuracy " + str(bad_hit_accuracy) + " innocent rate " + str(
        innocent_rate) + " appropriate accuracy " + str(appropriate_accuracy)


def train():
    train_x, test_x, train_y, test_y = prepare_data()
    n_dim = train_x.shape[1]

    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.ones([n_dim, 1]))

    init = tf.global_variables_initializer()

    learning_rate = 0.01
    training_epochs = 1000
    cost_history = np.empty(shape=[1], dtype=float)

    # this is variable deducted from existing variable, so we do not have to initialize it
    y_ = tf.matmul(X, W)
    cost = tf.reduce_mean(tf.square(y_ - Y))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)

    for epoch in range(training_epochs):
        sess.run(training_step, feed_dict={X: train_x, Y: train_y})
        cost_history = np.append(cost_history, sess.run(cost, feed_dict={X: train_x, Y: train_y}))
        print "epoch " + str(epoch) + " percentage " + str(float(epoch) / float(training_epochs) * 100) + "%"

    weights = extract_weights(sess, W)
    plt.plot(range(len(cost_history)), cost_history)
    plt.axis([0, training_epochs, 0, np.max(cost_history)])
    # plt.show()

    save_path = saver.save(sess, path_model)
    print("Model saved.")
    pred_y = sess.run(y_, feed_dict={X: test_x})

    measure_accuracy(test_y, pred_y)


train()


def test():
    if not os.path.isfile(path_model):
        return
    n_dim = dataset.TOTAL_LENGTH
    train_x, test_x, train_y, test_y = prepare_data()

    X = tf.placeholder(tf.float32, [None, n_dim])
    Y = tf.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.ones([n_dim, 1]))

    init = tf.global_variables_initializer()

    learning_rate = 0.01

    # this is variable deducted from existing variable, so we do not have to initialize it
    y_ = tf.matmul(X, W)
    cost = tf.reduce_mean(tf.square(y_ - Y))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)
    saver.restore(sess, path_model)

    pred_y = sess.run(y_, feed_dict={X: test_x})

    print("Model restored.")
