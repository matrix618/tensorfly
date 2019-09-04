#encoding:utf-8
import tensorflow as tf
 
# 为了简化过程，假设一个4×4×3的样本数据如下，
# 其中，第一个字符“0”表示图片的标签label
# “1”表示图片颜色值的R通道，“2”表示G通道，“3”表示B通道
source = '0111111111111111122222222222222223333333333333333'
sourcelist = tf.decode_raw(source, tf.uint8)

#上面运行后得到的数据如下：（0的ASCII值是48，同理推出1、2、3的值为49,50,51,这不是重点不用关心）
#[48 49 49 49 49 49 49 49 49 49 49 49 49 49 49 49 49 50 50 50 50 50 50 50
# 50 50 50 50 50 50 50 50 50 51 51 51 51 51 51 51 51 51 51 51 51 51 51 51
# 51]
 
#获取label
label = tf.strided_slice(sourcelist, [0], [1]);
 
#获取图片数据，并转为[3, 4, 4]的矩阵形式，其中，
#[1]表示从1下标开始截取，[49]表示截取到49下标，[3, 4, 4]中， 3表示通道数，4分别表示宽度和高度
image = tf.reshape(tf.strided_slice(sourcelist, [1], [49]), [3, 4, 4])
#上面运行后得到数据如下：
# [[[49 49 49 49]
#   [49 49 49 49]
#   [49 49 49 49]
#   [49 49 49 49]]
#
#  [[50 50 50 50]
#   [50 50 50 50]
#   [50 50 50 50]
#   [50 50 50 50]]
#
#  [[51 51 51 51]
#   [51 51 51 51]
#   [51 51 51 51]
#   [51 51 51 51]]]
#可以看到，RGB数据都分别在同一维度
 
#这里就是对上面得到的矩阵进行转置
image_transpose = tf.transpose(image, [1, 2, 0])
#上面运行后得到的数据如下
# [[[49 50 51]
#   [49 50 51]
#   [49 50 51]
#   [49 50 51]]
#
#  [[49 50 51]
#   [49 50 51]
#   [49 50 51]
#   [49 50 51]]
#
#  [[49 50 51]
#   [49 50 51]
#   [49 50 51]
#   [49 50 51]]
#
#  [[49 50 51]
#   [49 50 51]
#   [49 50 51]
#   [49 50 51]]]
 
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(tf.cast(sourcelist, tf.int32))
    print(result)
    result = sess.run(tf.cast(image, tf.int32))
    print(result)
    result = sess.run(tf.cast(image_transpose, tf.int32))
    print(result)
