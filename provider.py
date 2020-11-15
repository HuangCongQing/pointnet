import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification  下载点云分类数据集
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR): # 是否包含data数据集
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    # 加上--no-check-certificate 指令后即可进行下载
    os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))

# 在B这个维度上随机打乱数据。注释中输入维度B：batch_size， N：num_points
def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))     #arange创建等差数列，0到最大值，也就是labels的编号 
    np.random.shuffle(idx) # 随机打乱idx
    return data[idx, ...], labels[idx], idx # 打乱后点的idx

# 随机旋转点云  沿向上方向转
def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)  # 根据batch_data的矩阵结构，构造一个元素都是0的矩阵
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi # 设置旋转角度 #随机的一个转角
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])  #旋转矩阵
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
         #一个shape_pc内切成无数个3元素的数组
    return rotated_data

# 定角度旋转
def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

# bath中加入正态分布的噪声
def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip) #正负clip间的正态分布加到batch_data
    jittered_data += batch_data # 加入正太分布的噪声
    return jittered_data

# 下面函数都是文件信息
# 得到文件中每一行数据(此代码中每行数据为文件路径)
def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

# 加载h5文件
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label) # 返回数据和标签

# 加载h5文件
def loadDataFile(filename):
    return load_h5(filename)

# 加载带label和seg的h5数据
def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)

# 
def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)
