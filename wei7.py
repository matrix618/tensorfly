#encoding:utf-8
import tensorflow as tf
import os
import cifar10
import numpy as np
 
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
batch_size = 128
height = 24
width = 24
 
 
# 查看CIFAR-10数据是否存在，如果不存在则下载并解压
def download():
    # tf.app.flags.FLAGS是tensorflow的一个内部全局变量存储器
    FLAGS = tf.app.flags.FLAGS
    # 为了方便，我们将这个路径改为当前位置
    FLAGS.data_dir = './tmp/cifar10_data'
    # 如果不存在数据文件则下载，并且解压
    cifar10.maybe_download_and_extract()
 
#检测CIFAR-10数据是否存在，如果不存在则返回False
def check_cifar10_data_files(filenames):
    for file in filenames:
        if os.path.exists(file) == False:
            print('Not found cifar10 data.')
            return False
    return True
 
#获取每个样本数据，样本由一个标签+一张图片数据组成
def get_record(queue):
    print('get_record')
    #定义label大小，图片宽度、高度、深度，图片大小、样本大小
    label_bytes = 1
    image_width = 32
    image_height = 32
    image_depth = 3
    image_bytes = image_width * image_height * image_depth
    record_bytes = label_bytes + image_bytes
 
    #根据样本大小读取数据
    reader = tf.FixedLengthRecordReader(record_bytes)
    key, value = reader.read(queue)
 
    #将获取的数据转变成一维数组
    #例如
    # source = 'abcde'
    # record_bytes = tf.decode_raw(source, tf.uint8)
    #运行结果为[ 97  98  99 100 101]
    record_bytes = tf.decode_raw(value, tf.uint8)
 
    #获取label，label数据在每个样本的第一个字节
    label_data = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
 
    #获取图片数据，label后到样本末尾的数据即图片数据，
    # 再用tf.reshape函数将图片数据变成一个三维数组
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],[label_bytes + image_bytes]),
        [3, 32, 32])
 
    #矩阵转置，上面得到的矩阵形式是[depth, height, width]，即红、绿、蓝分别属于一个维度的，
    #假设只有3个像素，上面的格式就是RRRGGGBBB
    #但是我们图片数据一般是RGBRGBRGB，所以这里要进行一下转置
    #注：上面注释都是我个人的理解，不知道对不对
    image_data = tf.transpose(depth_major, [1, 2, 0])
 
    return label_data, image_data
 
def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  num_preprocess_threads = 1
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)
 
  # Display the training images in the visualizer.
  tf.summary.image('images', images)
 
  return images, tf.reshape(label_batch, [batch_size])
 
#获取图片前的预处理，检测CIFAR10数据是否存在，如果不存在直接退出
#如果存在，用string_input_producer函数创建文件名队列，
# 并且通过get_record函数获取图片标签和图片数据，并返回
def get_image(data_path):
    filenames = [os.path.join(data_path, "data_batch_%d.bin" % i) for i in range(1, 6)]
    print(filenames)
    if check_cifar10_data_files(filenames) == False:
        exit()
 
    #创建文件名队列
    queue = tf.train.string_input_producer(filenames)
    # 获取图像标签和图像数据
    label, image = get_record(queue)
    #将图像数据转成float32类型
    reshaped_image = tf.cast(image, tf.float32)
 
    #下面是数据增强操作
    #将图片随机裁剪成24*24像素
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
 
##    # 将图片随机左右翻转
##    distorted_image = tf.image.random_flip_left_right(distorted_image)
## 
##    #随机调整图片亮度
##    distorted_image = tf.image.random_brightness(distorted_image,
##                                                 max_delta=63)
##    #随机改变图片对比度
##    distorted_image = tf.image.random_contrast(distorted_image,
##                                               lower=0.2, upper=1.8)
    # 对图片标准化处理
    float_image = tf.image.per_image_standardization(distorted_image)
 
    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    label.set_shape([1])
 
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    return _generate_image_and_label_batch(float_image, label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)
 
#初始化过滤器
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
 
#初始化偏置，初始化时，所有值是0.1
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
 
#卷积运算，strides表示每一维度滑动的步长，一般strides[0]=strides[3]=1
#第四个参数可选"Same"或"VALID"，“Same”表示边距使用全0填充
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
 
#池化运算
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
 
if __name__ == '__main__':
    #查看CIFAR-10数据是否存在，如果不存在则下载并解压
    download()
 
    #获取图片数据
    images, labels = get_image('./tmp/cifar10_data/cifar-10-batches-bin/')
 
    # 创建x占位符，用于临时存放CIFAR10图片的数据，
    # [None, height , width , 3]中的None表示不限长度
    x = tf.placeholder(tf.float32, [None, height , width , 3])
    # y_存的是实际图像的标签，即对应于每张输入图片实际的值，
    # 为了方便对比，我们获得标签后，将起转成one-hot格式
    y_ = tf.placeholder(tf.float32, [None, 10])
 
 
    # 第一层卷积
    # 将过滤器设置成5×5×3的矩阵，
    # 其中5×5表示过滤器大小，3表示深度
    # 32表示卷积在经过每个5×5大小的过滤器后可以算出32个特征，即经过卷积运算后，输出深度为32
    W_conv1 = weight_variable([5, 5, 3, 32])
    # 有多少个输出通道数量就有多少个偏置
    b_conv1 = bias_variable([32])
    # 使用conv2d函数进行卷积计算，然后再用ReLU作为激活函数
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    # 卷积以后再经过池化操作
    h_pool1 = max_pool_2x2(h_conv1)
 
    # 第二层卷积
    # 因为经过第一层卷积运算后，输出的深度为32,所以过滤器深度和下一层输出深度也做出改变
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
 
    # 全连接层
    # 经过两层卷积后，图片的大小为6×6（第一层池化后输出为（24/2）×（24/2），
    # 第二层池化后输出为（12/2）×（12/2））,深度为64，
    # 我们在这里加入一个有1024个神经元的全连接层，所以权重W的尺寸为[6 * 6 * 64, 1024]
    W_fc1 = weight_variable([6 * 6 * 64, 1024])
    # 偏置的个数和权重的个数一致
    b_fc1 = bias_variable([1024])
    # 这里将第二层池化后的张量（长：6 宽：6 深度：64） 变成向量（跟上一节的Softmax模型的输入一样了）
    h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 6 * 64])
    # 使用ReLU激活函数
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
 
    # dropout
    # 为了减少过拟合，我们在输出层之前加入dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
    # 输出层
    # 全连接层输入的大小为1024,而我们要得到的结果的大小是10（0～9），
    # 所以这里权重W的尺寸为[1024, 10]
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    # 最后都要经过Softmax函数将输出转化为概率问题
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
 
    # 损失函数和损失优化
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv)))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
 
    # 测试准确率,跟Softmax回归模型的一样
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
    # 将训练结果保存，如果不保存我们这次训练结束后的结果也随着程序运行结束而释放了
    savePath = './tmp/mycifar_conv/'
    saveFile = savePath + 'mycifar_conv.ckpt'
    if os.path.exists(savePath) == False:
        os.mkdir(savePath)
    saver = tf.train.Saver()
 
    with tf.Session() as sess:
        #初始化变量
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners()
        for i in range(15000):
            label_batch, image_batch = sess.run([labels, images])
            label_batch_onehot = np.eye(10, dtype=float)[label_batch]
 
            sess.run(train_step, feed_dict={x:image_batch, y_:label_batch_onehot, keep_prob:1.0})
 
            if i % 100 == 0:
                result = sess.run(accuracy, feed_dict={x:image_batch, y_:label_batch_onehot, keep_prob:1.0})
                print('-----'+str(i))
                print(result)
 
        # 最后，将会话保存下来
        saver.save(sess, saveFile)
