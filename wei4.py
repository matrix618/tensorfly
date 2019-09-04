#coding:utf-8
#导入官方cifar10模块
import cifar10
import tensorflow as tf
 
#tf.app.flags.FLAGS是tensorflow的一个内部全局变量存储器
FLAGS = tf.app.flags.FLAGS
#cifar10模块中预定义下载路径的变量data_dir为'/tmp/cifar10_eval',预定义如下：
#tf.app.flags.DEFINE_string('data_dir', './cifar10_data',
#                           """Path to the CIFAR-10 data directory.""")
#为了方便，我们将这个路径改为当前位置
FLAGS.data_dir = './cifar10_data'
 
#如果不存在数据文件则下载，并且解压
cifar10.maybe_download_and_extract()
    
    



          
                              

 
