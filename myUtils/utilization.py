import os
import yaml
import torch
from scipy import sparse
from numpy import exp
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def angular_neighbors(vec, n):
    """
    Returns the indices of the n closest neighbors (excluding the vector itself)
    given an array of m points with x, y and z coordinates.

    Input : A m x 3 array, with m being the number of points, one per line.
    Each column has x, y and z coordinates for each vector.

    Output : A m x n array. Each line has the n indices of
    the closest n neighbors amongst the m input vectors.

    Note : Symmetries are not considered here so a vector and its opposite sign
    counterpart will be considered far apart, even though in dMRI we consider
    (x, y, z) and -(x, y, z) to be practically identical.
    """

    # Sort the values and only keep the n closest neighbors.
    # The first angle is always 0, since _angle always
    # computes the angle between the vector and itself.
    # Therefore we pick the rest of n+1 vectors and exclude the index
    # itself if it was picked, which can happen if we have N repetition of dwis
    # but want n < N angular neighbors

    # arr1 = np.argsort(_angle(vec))[:, :n+1]
    # arr3 = np.argsort(-_angle_cosine(vec))[:, :n + 1]

    arr = np.argsort(-abs_angle(vec))[:, :n + 1]
    #上面操作后返回n个最近邻居的索引

    # arr_angle = np.argsort(_angle(vec))
    # arr_cosine_similarity = np.argsort(-_cosine_similarity(vec))
    #
    # sum=np.sum(arr_angle==arr_cosine_similarity)
    # print(sum)

    # We only want n elements - either we remove an index and return the remainder
    # or we don't and only return the n first indexes.
    output = np.zeros((arr.shape[0], n), dtype=np.int32)
    for i in range(arr.shape[0]):
        cond = i != arr[i]
        output[i] = arr[i, cond][:n]

    return output

def _angle(vec):
    """
    Inner function that finds the angle between all vectors of the input.
    The diagonal is the angle between each vector and itself, thus 0 everytime.
    It should not be called as is, since it serves mainly as a shortcut for other functions.

    arccos(0) = pi/2, so b0s are always far from everyone in this formulation.
    """

    vec = np.array(vec)

    if vec.shape[1] != 3:
        raise ValueError("Input must be of shape N x 3. Current shape is {}".format(vec.shape))

    # Each vector is normalized to unit norm. We then replace
    # null norm vectors by 0 for sorting purposes.
    # Now each vector will have a angle of pi/2 with the null vector.
    # 每个向量被归一化为单位范数。然后我们更换
    # 空范数向量除以0进行排序。
    # 现在每个向量与空向量的夹角为π/2。

    with np.errstate(divide='ignore', invalid='ignore'):
        vec = vec / np.sqrt(np.sum(vec**2, axis=1, keepdims=True))
        vec[np.isnan(vec)] = 0
    #经上述运算后vec变为(a/sqrt(a**2+b**2+c**2),b/sqrt(a**2+b**2+c**2),c/sqrt(a**2+b**2+c**2)) 归一化

    angle = [np.arccos(np.dot(vec, v).clip(-1, 1)) for v in vec]
    a=np.array(angle)
    return np.array(angle)

def abs_angle(vec):
    return np.abs(_angle_cosine(vec))

def _angle_cosine(vec):
    vec = np.array(vec)
    if vec.shape[1] != 3:
        raise ValueError("Input must be of shape N x 3. Current shape is {}".format(vec.shape))
    angle=cosine_similarity(vec)
    return angle

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def prepare_sub_folder(output_directory):

    image_directory = os.path.join(output_directory, 'images')

    if not os.path.exists(image_directory):

        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)

    checkpoint_directory = os.path.join(output_directory, 'checkpoints')

    if not os.path.exists(checkpoint_directory):

        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)

    return checkpoint_directory, image_directory

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def convert(seconds):
    '''convert time format
    :param seconds:
    :return:
    '''
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return "%d:%02d:%02d" % (hour, min, sec)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape): #枚举【100，120】
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        # print("n：",n)
        r = (v1 - v0) / (2 * n)  # 2/2*i
        # print("r",r)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        # print("seq",seq)
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    # print("img.shape[-2:]",img) # [8 100, 120]
    coord = make_coord(img.shape[-2:])
    # rgb = img.view(8, -1).permute(1, 0) # 如果写成permute（1,0）得到的就是矩阵的转置
    graf=[]
    n ,m = img.shape[-2:]
    # print(n,m)
    for i in range(0,n):
        for j in range(0,m):
            # print("{},{}:".format(i,j),img[...,i,j])
            graf.append(np.array(img[...,i,j].cpu()))
    # print("graf.shape",np.array(graf).shape)
    graf = rearrange(graf, "n b h  -> b n h")
    return coord, graf

def weight_to_one(weight):
    weight=[i/(weight.sum()+1e-8) for i in weight]
    return torch.tensor(weight)

import math
def getalike(i,bvecs,bvals,num):
    '''
      w=exp{-(1-(q1*q2)**2)/2a1**2}exp{-(sqrt(b1)-sqrt(b2))**2/2a2**2}
      a1=0.26,a2=0.1
    '''
    # num=24
    ids = math.cos((math.pi)/num)
    a1 = math.sqrt(1 -ids*ids)

    # a1 = 0.21
    # a1 = 0.59
    # a11 = np.sqrt(1-np.cos(15))
    a2 = 1
    qk = bvecs[i]
    bk = bvals[i]
    fp = (a1**2) * 2
    tp = ((a2**2) *2)
    # print("fp:",fp)
    # print("tp:", fp)
    b = 1 - np.dot(bvecs,qk)**2
    # print('b:',b)
    t1 = exp(-b * 1.0/fp)
    # print('t1:', t1)
    c = np.sqrt(bk) - np.sqrt(bvals)
    # print('c:', c)
    t2 = exp(-(c**2)*1.0 /tp)
    # print('t2:', t2)

    l = len(t2)
    for j in range(l):
        t2[j]= t2[j] * t1[j]
    weight = torch.tensor([i / (t1.sum() + 1e-9) for i in t1],dtype=torch.float32)

    return t1.T

def getAdm(bvecs,bvals,num_node,num):

    A = np.zeros((num_node,num_node))
    for i in range(num_node):
        A[i:i+1,...] = getalike(i,bvecs,bvals,num)
    return A

def get_Gft(bvals,bvecs,FrameType,num):

    vecs = bvecs
    vals = bvals
    # print('vecs:',vecs.shape)
    num_node = len(vecs)
    A = getAdm(vecs, vals, num_node,num)
    return A


