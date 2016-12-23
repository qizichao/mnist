# MNIST using Linear + Softmax

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Download and load MNIST data.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True);

x = tf.placeholder(tf.float32, shape = [None, 784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#global_variables_initializers
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch = mnist.train.next_batch(100)
	train.run(session=sess, feed_dict={x: batch[0], y_: batch[1]})


# Evaluate the model.
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels})