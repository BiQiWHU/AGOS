
from my_ops import *
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow as tf
import math

def AGOS(conv5, reuse=False, is_training=True):
    with tf.variable_scope('mil', reuse=reuse) as scope:
            conv6_0 = conv(conv5, 1, 1, 256, 1, 1, name='conv6_0', bn=True, is_training=is_training)
            conv6_1 = dilated_conv(conv5, 3, 3, 256, 1, name='conv6_1', bn=True, is_training=is_training)
            conv6_3 = dilated_conv(conv5, 3, 3, 256, 3, name='conv6_3', bn=True, is_training=is_training)
            conv6_5 = dilated_conv(conv5, 3, 3, 256, 5, name='conv6_5', bn=True, is_training=is_training)
            conv6_7 = dilated_conv(conv5, 3, 3, 256, 7, name='conv6_7', bn=True, is_training=is_training)       
#            conv6_9 = dilated_conv(conv5, 3, 3, 256, 9, name='conv6_9', bn=True, is_training=is_training)

            ### differential convolutional profile
            dc1 = tf.abs(conv6_1 - conv6_0)
            dc2 = tf.abs(conv6_3 - conv6_1)
            dc3 = tf.abs(conv6_5 - conv6_3)
            dc4 = tf.abs(conv6_7 - conv6_5)
#            dc5=tf.abs(conv6_9-conv6_7)

            ### multi-scale MIL
            conv6_0 = conv(conv6_0, 1, 1, 256, 1, 1, name='conv7_1', bn=True, is_training=is_training)
            conv6_0 = conv(conv6_0, 1, 1, 30, 1, 1, name='conv8_1', relu=False)
            
            conv6_0_ = tf.reduce_sum(conv6_0, [1, 2])

            dc1 = conv(dc1, 1, 1, 256, 1, 1, name='conv7_2', bn=True, is_training=is_training)
            dc1 = conv(dc1, 1, 1, 30, 1, 1, name='conv8_2', relu=False)
            dc1_ = tf.reduce_sum(dc1, [1, 2])

            dc2 = conv(dc2, 1, 1, 256, 1, 1, name='conv7_3', bn=True, is_training=is_training)
            dc2 = conv(dc2, 1, 1, 30, 1, 1, name='conv8_3', relu=False)
            dc2_ = tf.reduce_sum(dc2, [1, 2])

            dc3 = conv(dc3, 1, 1, 256, 1, 1, name='conv7_4', bn=True, is_training=is_training)
            dc3 = conv(dc3, 1, 1, 30, 1, 1, name='conv8_4', relu=False)
            dc3_ = tf.reduce_sum(dc3, [1, 2])
#
            dc4 = conv(dc4, 1, 1, 256, 1, 1, name='conv7_5', bn=True, is_training=is_training)
            dc4 = conv(dc4, 1, 1, 30, 1, 1, name='conv8_5', relu=False)
            dc4_ = tf.reduce_sum(dc4, [1, 2])
#            
#            dc5 = conv(dc5, 1, 1, 256, 1, 1, name='conv7_6', bn=True, is_training=is_training)
#            dc5 = conv(dc5, 1, 1, 30, 1, 1, name='conv8_6', relu=False)
#            dc5_ = tf.reduce_sum(dc5, [1, 2])
#            
            #### residual fusion
            conv6_=conv6_0_ + dc1_ + dc2_ + dc3_ + dc4_ # +dc5_
            #conv6_=dc1_+dc2_+dc3_+dc4_
            
            #### self-aligned module
            semantic_rest=tf.abs(dc1-conv6_0)+tf.abs(dc2-conv6_0)+tf.abs(dc3-conv6_0)+tf.abs(dc4-conv6_0)#+tf.abs(dc5-conv6_0)
            semantic_rest = tf.reduce_sum(semantic_rest, [1, 2])

            ### avr MIL pooling
            instance_pooling = tf.layers.batch_normalization(conv6_, training=is_training, momentum=0.999)
            semantic_rest = tf.layers.batch_normalization(semantic_rest, training=is_training, momentum=0.999)
        
            return instance_pooling, semantic_rest, conv6_
