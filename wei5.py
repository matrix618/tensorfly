#encoding:utf-8
import tensorflow as tf
import matplotlib.pyplot as plt
 
#读取原始图像数据
image_data = tf.gfile.GFile('./cifar10_image/baboon.jpg', 'rb').read()
 
with tf.Session() as sess:
    #对图像使用jpg格式解码，得到三维数据
    pltdata = tf.image.decode_jpeg(image_data)

    pltdata = tf.image.flip_up_down(pltdata)

    #显示图像
    plt.imshow(pltdata.eval())
    plt.show()
