import tensorflow as tf
import numpy as np

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x=tf.placeholder(tf.float32,[None,784])

w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

y=tf.nn.softmax(tf.matmul(x,w)+b)

