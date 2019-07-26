import tensorflow as tf
import numpy as np

m1 = tf.constant([[1.,2.]])

with tf.Session() as sess:
    result = sess.run(m1)
    print(result)
