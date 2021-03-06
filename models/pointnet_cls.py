#coding=utf-8
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net # 引用T-Net

# 根据shape向pointclouds_pl, labels_pl中添加float32和int32的占位符   添加float和int的占位符，让pointcloud_pl, label_pl形式符合batchsize
def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3)) # B*N*3
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size)) # B
    return pointclouds_pl, labels_pl

# 一层一层得到网络结构（INPUT BxNx3  Output Bx40）
def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value  # 32
    num_point = point_cloud.get_shape()[1].value   # 1024个点
    end_points = {}  # 定义

     # 创建一个命名空间，名字为：transform_net1。然后在作用域下定义一个变量 transform(该变量可以在后面使用)
    # tf.variable_scope(<scope_name>) 必须要在tf.variable_scope的作用域下使用tf.get_variable()函数
    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)  # shape（32， 3， 3） 输入转换
    
    # 两个三维矩阵的乘法怎样计算呢?我通过实验发现，tensorflow把前面的维度当成是batch，对最后两维进行普通的矩阵乘法。 
    # 也就是说，最后两维之前的维度，都需要相同。
    point_cloud_transformed = tf.matmul(point_cloud, transform)  # 矩阵相乘shape（32， 1024， 3） =   shape（32， 1024， 3）  *   shape（32， 3， 3）
    #通过T-net网络
    input_image = tf.expand_dims(point_cloud_transformed, -1)  #扩展成 4D 张量，在最后增加一维： shape（32， 1024， 3， 1）  -1代表最后一维度

    net = tf_util.conv2d(input_image, 64, [1,3],   # 64 ：  num_output_channels  #  shape（32， 1024， 1， 64）
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],      #  shape（32， 1024， 1， 64）
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
#tf_util包中conv2d，首先进入的是input_image，然后连接到下一个net
    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64) # 特征转化
    end_points['transform'] = transform # end_points（32，64，64）  #end_points 用于存储张量 transform 的信息。是一个字典？
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform) #  shape（32， 1024， 64） # 通过指定 axis 来删除特定的大小为1的维度
    net_transformed = tf.expand_dims(net_transformed, [2])  #  shape（32， 1024，1，  64）

    net = tf_util.conv2d(net_transformed, 64, [1,1],  # #  shape（32， 1024，1，  64）
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],  # #  shape（32， 1024，1，  128)
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],  # #  shape（32， 1024，1，  1024)
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
#再接着进入conv345
    # Symmetric function: max pooling 最大池化
    net = tf_util.max_pool2d(net, [num_point,1],  #  结果 shape（32， 1 ，1，  1024)
                             padding='VALID', scope='maxpool')
#conv5进入pool
    # 定义分类的mpl512-256-k, k为分类类别数目
    #   # PointNet利用了一个三层感知机MPL(512–256–40)来对特征进行学习，最终实现了对于40类的分类.
    net = tf.reshape(net, [batch_size, -1])  # reshape转化为二维   shape（32， 1024)
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,  #    shape（32， 512)
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,  
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, #    shape（32， 256)
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')  #    shape（32， 40)

    return net, end_points  # 返回

# 总loss=classify_loss + mat_diff_loss * reg_weight
def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))  # 输入
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
