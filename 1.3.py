import tensorflow as tf
import numpy as np

#####################################################
m1 = tf.constant([[1.,2.]])
m2 = tf.constant([[3.],[5.]])

with tf.Session() as sess:
    with tf.device("/gpu:0"):
        result = sess.run(m1)
        print(result)
        result = sess.run(m2)
        print(result)
        m3 = tf.matmul(m1,m2)
        result = sess.run(m3)
        print(result)

###################################################
# 创建一个变量, 初始化为标量 0.
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init = tf.global_variables_initializer()

# 启动图, 运行 op
with tf.Session() as sess:
  # 运行 'init' op
  sess.run(init)
  # 打印 'state' 的初始值
  print(sess.run(state))
  # 运行 op, 更新 'state', 并打印 'state'
  for _ in range(3):
    sess.run(update)
    print(sess.run(state))

###################################################
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
  result = sess.run([mul, intermed])
  print(result)

# 输出:
# [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]

###################################################
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    result = sess.run(output,feed_dict={input1:[5],input2:[3]})
    print(result)
    
    


