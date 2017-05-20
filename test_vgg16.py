import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import vgg16
import utils
def reshape_layer( bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[ [batch_size], [num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf   

def softmax_layer( bottom, name):
    if name == 'rpn_cls_prob_reshape':
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_layer,
                                    [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                     self._feat_stride, self._anchors, self._num_anchors],
                                    [tf.float32, tf.float32])
      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])

    return rois, rpn_scores
img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

#batch1 = img1.reshape((1, 224, 224, 3))
#batch2 = img2.reshape((1, 224, 224, 3))
#batch = np.concatenate((batch1, batch2), 0)
batch= img1.reshape((1, 224, 224, 3))

batch_size=2
num_classes = 2
anchor_scales=(8, 16, 32)
anchor_ratios=(0.5, 1, 2)
num_scales = len(anchor_scales)
num_ratios = len(anchor_ratios)
num_anchors = num_scales * num_ratios

is_training=True
initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
images = tf.placeholder("float", [1, 224, 224, 3])
feed_dict = {images: batch}

with tf.Session(  
    config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:

    vgg = vgg16.Vgg16()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

#    prob = sess.run(vgg.prob, feed_dict=feed_dict)
    net= sess.run(vgg.pool5 , feed_dict=feed_dict)
    
    
    rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training,
                      weights_initializer=initializer)
#    self._act_summaries.append(rpn)
    rpn_cls_score = slim.conv2d(rpn, num_anchors * 2, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None)
      # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
    rpn_cls_prob_reshape = softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
    rpn_cls_prob = reshape_layer(rpn_cls_prob_reshape, num_anchors * 2, "rpn_cls_prob")
    rpn_bbox_pred = slim.conv2d(rpn, num_anchors * 4, [1, 1], trainable=is_training,
                                  weights_initializer=initializer,
                                  padding='VALID', activation_fn=None)
    rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
    rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
print ('done here) 

#    print(prob)
#    print(pool5)
#    utils.print_prob(prob[0], './synset.txt')
#    utils.print_prob(prob[1], './synset.txt')
