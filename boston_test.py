import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from numpy import genfromtxt
from sklearn.datasets import load_boston


def read_boston_data():
    boston = load_boston()
    features = np.array(boston.data)
    labels = np.array(boston.target)
    return features, labels


def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma


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


# load features and labels 504x13, 504x1
features, labels = read_boston_data()
# normalize features to 0-1
normalized_features = feature_normalize(features)
# append bias to features and labels
f, l = append_bias_reshape(normalized_features, labels)
# number of dimensions (14)
n_dim = f.shape[1]

# randomly chose 80% of data as train, 20% as test
rnd_indices = np.random.rand(len(f)) < 0.80

train_x = f[rnd_indices]
train_y = l[rnd_indices]
test_x = f[~rnd_indices]
test_y = l[~rnd_indices]

learning_rate = 0.01
training_epochs = 1000
cost_history = np.empty(shape=[1], dtype=float)

# during the initialize part, data will not be fed in
# we only initialize variables
# Here None means that a dimension can be of any length.
# so the size of X is (None x n_dim)
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.ones([n_dim, 1]))

init = tf.initialize_all_variables()

# this is variable deducted from existing variable, so we do not have to initialize it
y_ = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(y_ - Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={X: train_x, Y: train_y})
    cost_history = np.append(cost_history, sess.run(cost, feed_dict={X: train_x, Y: train_y}))

plt.plot(range(len(cost_history)), cost_history)
plt.axis([0, training_epochs, 0, np.max(cost_history)])
plt.show()

pred_y = sess.run(y_, feed_dict={X: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))


fig, ax = plt.subplots()
ax.scatter(test_y, pred_y)
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()