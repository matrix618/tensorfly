# coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
 
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))
 
y = tf.nn.softmax(tf.matmul(x, w) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
 
saver = tf.train.Saver();
 
with tf.Session() as sess:

    saver.restore(sess, './saver/mnist.ckpt')
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images[:10], y_: mnist.test.labels[:10]}))

    print(sess.run(y_, feed_dict={x: mnist.test.images[:10], y_: mnist.test.labels[:10]}))
    print(sess.run(y, feed_dict={x: mnist.test.images[:10], y_: mnist.test.labels[:10]}))

    print(sess.run(tf.argmax(y_, 1), feed_dict={x: mnist.test.images[:10], y_: mnist.test.labels[:10]}))
    print(sess.run(tf.argmax(y, 1), feed_dict={x: mnist.test.images[:10], y_: mnist.test.labels[:10]}))



##    print(sess.run(y))


    
    



          
                              

 
