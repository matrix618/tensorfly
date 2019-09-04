#encoding:utf-8
import tensorflow as tf
import os
import cifar10
import scipy.misc
 
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
 
#获取图片前的预处理，检测CIFAR10数据是否存在，如果不存在直接退出
#如果存在，用string_input_producer函数创建文件名队列，
# 并且通过get_record函数获取图片标签和图片数据，并返回
def get_image(data_path):
    filenames = [os.path.join(data_path, "data_batch_%d.bin" % i) for i in range(1, 6)]
    print(filenames)
    if check_cifar10_data_files(filenames) == False:
        exit()
    queue = tf.train.string_input_producer(filenames, shuffle=False)
    # return tf.cast((cifar10_input.read_cifar10(queue)).uint8image, tf.float32)
    return get_record(queue)
 
 
if __name__ == '__main__':
    #查看CIFAR-10数据是否存在，如果不存在则下载并解压
    download()
 
    #将获取的图片保存到这里
    image_save_path = './cifar10_image/'
    if os.path.exists(image_save_path) == False:
        os.mkdir(image_save_path)
 
    #获取图片数据
    key, value = get_image('./tmp/cifar10_data/cifar-10-batches-bin/')


 
    with tf.Session() as sess:
        #初始化变量
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        #这里才真的启动队列
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(100):
            # print("i:%d" % i)
            ####################################
            #这里data和label不能分开run，否则图片和标签就不匹配了，多谢网友ATPY提醒
            #data = sess.run(value)
            #label = sess.run(key)
            #应该这样
            label, data = sess.run([key, value])
            ####################################
##            print(label)
##            scipy.misc.toimage(data).save(image_save_path + '/%d_%d.jpg' % (label, i))
        coord.request_stop()
        coord.join()

        print("ok")
