import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.rand(2,10))
y_data = np.dot([0.1,0.2],x_data)+0.3

print(x_data)
print(y_data)

# 构造一个线性模型 
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b


# 初始化变量
#将init = tf.initialize_all_variables() # old api 改为
init = tf.global_variables_initializer() #new api

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

print(sess.run(b))
print(sess.run(W))
print(sess.run(y))
print('--')
print(sess.run(y - y_data))
print(sess.run(tf.square(y - y_data)))
print('--')
# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
print(sess.run(loss))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)



# 拟合平面
for step in range(0, 1000):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(W), sess.run(b),sess.run(loss))
