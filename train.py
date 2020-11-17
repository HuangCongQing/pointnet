#coding=utf-8
import argparse # 程序使用端口指令
import math
import h5py
import numpy as np
import tensorflow as tf
import socket # 通信
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # '/home/hcq/pointcloud/pointnet'
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
print(sys.path) # sys.path是个数组
import provider
import tf_util

# 读取命令参数  
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]') # epoch    default: 250
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args() # 

# 参数设置
BATCH_SIZE = FLAGS.batch_size #  batch_size default=32
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch  # epoch    default: 250
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) #动态导入模型 import network module  # '--model', default='pointnet_cls'
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py') # 模型py文件
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def   复制文件到...
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')  # log存储路径
LOG_FOUT.write(str(FLAGS)+'\n') # FLAGS：Namespace(batch_size=32, decay_rate=0.7, decay_step=200000, .......

MAX_NUM_POINT = 2048  # 最大点数
NUM_CLASSES = 40   # 分类数

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()  # 获取本地主机名  # hcq-G5-5590

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))# 训练
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt')) # 测试

# log记录函数 #日志记录函数
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n') # 写入
    LOG_FOUT.flush()
    print(out_str)

# 获取学习率参数（学习率不断衰减）
def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
	# 训练时学习率最好随着训练衰减，learning_rate最大取0.00001
    return learning_rate        

# 获取Batch Normalization参数（认为限制最大0.00001）
#  BN会对每一个mini-batch数据的内部进行标准化（normalization）,使输出规范到N（0，1）的正态分布，加快了网络的训练速度,还可以增大学习率。
def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay( # tf.train.exponential_decay 指数衰减法
                      BN_INIT_DECAY, # 0.5
                      batch*BATCH_SIZE, # batch*batch_size
                      BN_DECAY_DECAY_STEP,  # 200000  衰减速度
                      BN_DECAY_DECAY_RATE,# 0.5   学习率衰减系数，通常介于0-1之间。
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)  # 0.99
    return bn_decay

# 调用 训练功能函数  和 评估功能函数
def train():
    with tf.Graph().as_default():  # 将这个类实例，运行环境的默认图，如果只有一个主线程不写也没有关系
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT) #models/pointnet_cls.py  pointclouds_pl: shape(32, 1024, 3)    label_pl: shape=(32, ) def placeholder_inputs(batch_size, num_point):
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl )  # 输出
            
            # Note the global_step=batch parameter to minimize.  # globe_step初始化为0，每次自动加1
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)  #创建一个变量,初始化为 0
            bn_decay = get_bn_decay(batch) # 批训练时， 得到 batch 的衰减率
            tf.summary.scalar('bn_decay', bn_decay) #衰减  tf.summary.scalar(),用于收集标量信息

            # Get model and loss 
            # 创建的数据处理网络为pred，调用 model\pointnet_cls 下的get_model()得到。由get_model()可知，pred的维度为B×N×40，40为分出的类别
            # Channel数，对应40个分类标签。每个点的这40个值最大的一个的下标即为所预测的分类标签。
            # 首先使用共享参数的MLP对每个点进行特征提取，再使用MaxPooling在特征维进行池化操作，使得网络对不同数量点的点云产生相同维度的特征向量，
            # 且输出对输入点的顺序产生不变性。在得到固定维度的特征向量之后，再使用一个MLP对其进行分类
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay) # pred（32，40）end_points（32，64，64）（INPUT BxNx3  Output Bx40）
            loss = MODEL.get_loss(pred, labels_pl, end_points) #  # 调用pointnet_cls下的get_loss（）
            tf.summary.scalar('loss', loss)#代价

            # # tf.argmax(pred, 2) 返回pred C 这个维度的最大值索引返回相同维度的bool值矩阵
            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl)) # tf.equal() 比较两个张量对应位置是否相等
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE) # BATCH_SIZE： 32
            tf.summary.scalar('accuracy', accuracy)#精度

            # Get training operator #获得衰减后的学习率，以及选择优化器optimizer
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':  # 默认adam
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            # minimize的内部存在两个操作：(1)计算各个变量的梯度 (2)用梯度更新这些变量的值
            # (1)计算loss对指定val_list的梯度（导数），返回元组列表[(gradient,variable),…]
            # (2)用计算得到的梯度来更新对应的变量（权重）
            # 注意：在程序中global_step初始化为0，每次更新参数时，自动加1
            # 将minimize()分成两个步骤的原因：在某种情况下对梯度进行修正，防止梯度消失或者梯度爆炸

            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()  # 保存所有变量
            
        # Create a session  #配置session 运行参数。
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # =True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用
        config.allow_soft_placement = True #当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
        config.log_device_placement = False  #在终端打印出各项操作是在哪个设备上运行的
        sess = tf.Session(config=config)   # 创建 sess, 才能运行框架

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), # 将训练过程数据保存在filewriter指定的文件中 tf.summary.FileWritter(path,sess.graph)
                                  sess.graph)  # 
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables # Init variables #初始化参数，开始训练
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        # ops 是一个字典，作为接口传入训练和评估 epoch 循环中。
        #  pred 是数据处理网络模块；loss 是 损失函数；train_op 是优化器；batch 是当前的批次
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op, # ???
               'merged': merged,
               'step': batch}
        # 大头在这里===============================================================================
        for epoch in range(MAX_EPOCH):  # 250个epoch
            log_string('**** EPOCH %03d ****' % (epoch))  #LOG信息
            sys.stdout.flush() ## 在Linux系统下，必须加入sys.stdout.flush()才能一秒输一个数字，不然，只能程序执行完之后才会一次性输出
             
            train_one_epoch(sess, ops, train_writer) # 训练功能函数
            eval_one_epoch(sess, ops, test_writer) # 评估功能函数
            
            # Save the variables to disk.
            if epoch % 10 == 0:  # 10次一循环 #每10个epoch保存1次模型
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))  # train_one_epoch 函数用来训练一个epoch
                log_string("Model saved in file: %s" % save_path) # eval_one_epoch函数用来每运行一个epoch后evaluate在测试集的accuracy和loss


# 训练功能函数（加载h5文件）
def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files 打乱训练文件
    train_file_idxs = np.arange(0, len(TRAIN_FILES)) # 5行 h5文件路径
    np.random.shuffle(train_file_idxs)
    
    for fn in range(len(TRAIN_FILES)):  # 循环5次
        log_string('----' + str(fn) + '-----')  # ----0-----
        # current_data: shape(2048, 2048, 3)     current_label:  shape(2048, 1)
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]]) # 加载h5文件  返回数据和标签
        current_data = current_data[:,0:NUM_POINT,:]  # NUM_POINT default=1024  shape(2048, 1024, 3)
        #[楼层,行,列] 三维数组
		#所有楼层  num行   所有列元素传给current_data
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]  #  file_size: 2048
        num_batches = file_size // BATCH_SIZE   # 整除，计算一共有多少个批次   计算在指定BATCH_SIZE下，训练1个epoch 需要几个mini-batch训练。

        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        # 在一个epoch 中逐个mini-batch训练直至遍历完一遍训练集。计算总分类正确数total_correct和已遍历样本数
        # total_senn，总损失loss_sum

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE # 起始idx
            end_idx = (batch_idx+1) * BATCH_SIZE # 
            
            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :]) #调用provider中rotate_point_cloud
            jittered_data = provider.jitter_point_cloud(rotated_data)#调用provider中jitter_point_cloud
            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            #  训练，使用 tf 的 session 运行设计的框架，ops['pred'] 为整个网络，feed_dict 为网络提供的数据
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val

        # 记录平均loss，以及平均accuracy。
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))

# 评估功能函数（测试）
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    for fn in range(len(TEST_FILES)): # 测试文件  2个h5文件
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE  # 整除，计算一共有多少个批次
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            #估计
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen))) # 
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
         


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
