#coding=utf-8
import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util  # pointnet/utils/tf_util.py

''' 
首先我们来看T-Net模型的代码，它的主要作用是学习出变化矩阵来对输入的点云或特征进行规范化处理。其中包含两个函数，分别是

学习点云变换矩阵的:input_transform_net(point_cloud, is_training, bn_decay=None, K=3)

学习特征变换矩阵的:feature_transform_net(inputs, is_training, bn_decay=None, K=64)
 '''


def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    # 转为4D张量，尺寸索引轴从零开始; 如果您指定轴的负数,则从最后向后计数
    ''' 
    tf.expand_dims(input, dim, name=None)
    e.g.
    ‘t2’ is a tensor of shape [2, 3, 5]
    shape(expand_dims(t2, 0)) --> [1, 2, 3, 5]
    shape(expand_dims(t2, 2)) --> [2, 3, 1, 5]
    shape(expand_dims(t2, 3)) --> [2, 3, 5, 1]
    '''
    
    input_image = tf.expand_dims(point_cloud, -1)
    # 构建T-Net模型，64--128--1024
    # 其中，tf_util.conv2d是作者自己写的函数，函数声明如下
    # 利用1*1的卷积来实现全连接。每一层单元依次为64-128-1024-512-256的网络结构
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')
    # 利用1024维特征生成256维度的特征
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)
    # #生成点云旋转矩阵 T=3*3
    with tf.variable_scope('transform_XYZ') as sc:
        assert (K == 3)
        # 通过定义权重[W(256,3K), bais(3K)]，将上面的256维特征转变为3*3的旋转矩阵输出。
        weights = tf.get_variable('weights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1,0,0,0,1,0,0,0,1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform


''' 同样对于特征层的规范化处理，其输入为n64的特征输出为6464的旋转矩阵，网络结构与上面完全相同，
只是在输入输出的维度需要变化：
 '''
def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value
	#构建T-Net模型，64--128--1024
    net = tf_util.conv2d(inputs, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    # 最大池化，二维的池化函数对点云中点的数目这个维度进行池化，n-->1
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)
    #生成特征旋转矩阵 T=64*64
    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform
