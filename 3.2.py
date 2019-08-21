import tensorflow as tf
import numpy as np

import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x=tf.placeholder(tf.float32,[None,784])

w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

y=tf.nn.softmax(tf.matmul(x,w)+b)

y_=tf.placeholder(tf.float32,[None,10])

cross_=-tf.reduce_sum(y_*tf.log(y))

train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

      if i%100==0:
          print("======="+str(i)+"=======")
          correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
          accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
          ys=sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels})
          print(ys)

          save_path = saver.save(sess, "tmp/model.ckpt")
          print("Model saved in file: ", save_path)



          
                              

 
