import numpy as np
import tensorflow as tf

import vgg16
import utils
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
    images = tf.placeholder("float", [2, 224, 224, 3])
    feed_dict = {images: batch}

    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    prob = sess.run(vgg.prob, feed_dict=feed_dict)
    pool5= sess.run(vgg.pool5, feed_dict=feed_dict)
    rpn = slim.conv2d(pool5, 512, [3, 3], trainable=True, weights_initializer=initializer)
    
    
#    print(prob)   
#    utils.print_prob(prob[0], './synset.txt')
#    utils.print_prob(prob[1], './synset.txt')
