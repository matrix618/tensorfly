from tensorflow.examples.tutorials.mnist import input_data
import os
import scipy.misc as sm
import numpy as np
 
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
save_dir = 'F:/tmp/image/'
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
 
for i in range(50):
    image_array = mnist.train.images[i, :]
    one_hot_label = mnist.train.labels[i, :]
    label = np.argmax(one_hot_label)
    image_array = image_array.reshape(28, 28)
    filename = save_dir + 'image_train_%d_%d.jpg' % (i, label)
    sm.toimage(image_array).save(filename)
    
    



          
                              

 
