#coding:utf-8
#导入官方cifar10模块
import cifar10
import tensorflow as tf
import os


#检测CIFAR-10数据是否存在，如果不存在则返回False
def check_cifar10_data_files(filenames):
    for file in filenames:
        if os.path.exists(file) == False:
            print('Not found cifar10 data.')
            return False
    return True
 
#获取图片前的预处理，检测CIFAR10数据是否存在，如果不存在直接退出
#如果存在，用string_input_producer函数创建文件名队列，
# 并且通过get_record函数获取图片标签和图片数据，并返回
def get_image(data_path):
    filenames = [os.path.join(data_path, "data_batch_%d.bin" % i) for i in range(1, 6)]
    print(filenames)
    if check_cifar10_data_files(filenames) == False:
        exit()
    queue = tf.train.string_input_producer(filenames)
    return get_record(queue)
 
#tf.app.flags.FLAGS是tensorflow的一个内部全局变量存储器
FLAGS = tf.app.flags.FLAGS
#cifar10模块中预定义下载路径的变量data_dir为'/tmp/cifar10_eval',预定义如下：
#tf.app.flags.DEFINE_string('data_dir', './cifar10_data',
#                           """Path to the CIFAR-10 data directory.""")
#为了方便，我们将这个路径改为当前位置
FLAGS.data_dir = './cifar10_data'
 
#如果不存在数据文件则下载，并且解压
cifar10.maybe_download_and_extract()

#将获取的图片保存到这里
image_save_path = './cifar10_image/'
if os.path.exists(image_save_path) == False:
    os.mkdir(image_save_path)


          
                              

 
