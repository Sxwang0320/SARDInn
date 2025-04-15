from torch.utils.data.dataset import Dataset
import numpy as np
import h5py

import torch
import torch.utils.data
from einops import rearrange
from myUtils.torchutils import get_uniform_downsample_indices
from myUtils.utilization import angular_neighbors,get_Gft
from matplotlib import pyplot as plt
import torchvision.transforms as transforms




class getTrainData(Dataset):
    '''
    Dataloader for the proposed framework with q-space augmentation
    '''
    def __init__(self, folder, mode='train',num_grad=90,num_slice=72,pad_size=(72,80),config=None):
        self.mode = mode
        self.config = config
        self.pad_size = pad_size
        self.grad_able = 20
        self.num_slice = num_slice
        self.num_grad = num_grad
        self.num_neb = 3
        self.key = self.mode
        # if self.mode=='train':
        #     self.key = 'train'
        # elif self.mode=='test':
        #     print("the main of getTrainData's method can't be applied to Test!")
        # elif self.mode == 'val':
        #     print("the main of getTrainData's method can't be applied to Val!")

        self.folder = folder
        self.ran=0
        self.dataset = 'HCP'

        print(self.dataset, 'numbel of num_grad: {}'.format(self.num_grad))
        hf = h5py.File(self.folder, 'r')
        # 最大b值
        self.bval_max = (hf['{}bvec'.format(self.mode)][:,: ,0]).max()
        hf.close()
        print(self.dataset, 'bval max: {}'.format(self.bval_max))

    def __getitem__(self, item):

        idx_slice,id_grad = self.ran // 6,self.ran % 6  # the order of slice and grad
        self.ran = self.ran + 1  # 更新ran
        hf = h5py.File(self.folder, 'r')

        slice_all = hf['{}'.format(self.key)][idx_slice]  # get the data from dataset

        # 当前实例的所有方向
        cond_all = hf['{}bvec'.format(self.key)][idx_slice]
        cond_list = get_uniform_downsample_indices(cond_all[:, 1:], 9,0) # 输出选择的点在原始点集中的索引,第一次下采样的索引，用于自监督
        neb_vec = get_uniform_downsample_indices(cond_all[cond_list, 1:], 3,0) # 输出选择的点在原始点集中的索引，第二次下采样，用于训练
        neb_vec = [cond_list[val] for i,val in enumerate(neb_vec)] # 取出这10个点在cond_all中的索引

        # q空间两次下采样均匀采样
        slice_neb = slice_all[neb_vec] # 训练集中的邻居节点

        pre_list = list(set(cond_list).difference(neb_vec)) # 需要预测的索引

        self.pre_id = pre_list[id_grad]
        # print("pre_id: ",self.pre_id)
        slice_pre = slice_all[self.pre_id]  # 训练时的预测节点

        # 裁剪图像

        # slice_neb = slice_neb[:, 10:self.pad_size[0] + 10, 20:self.pad_size[1] + 20]
        # slice_pre = slice_pre[10:self.pad_size[0] + 10, 20:self.pad_size[1] + 20]

        tmp = np.expand_dims(slice_pre, axis=0) # 扩展维度
        slice_neb = np.row_stack((tmp, slice_neb))  # 姜预测的方向在扩展的维度上加上已知方向

        tmp_vec = cond_all[self.pre_id] # 预测方向

        neb_vec = cond_all[neb_vec] # 邻居方向


        neb_vec = np.row_stack((tmp_vec, neb_vec)) # 组成邻接点和目标节点集

        # 预测的切片的位置
        index_slice = (64 + (idx_slice % self.num_slice)) * 0.01


        bvals = neb_vec[:, 0] * 1.0 / self.bval_max  # b值归一化
        bvecs = neb_vec[:, 1:]

        # 初步邻居节点，这里的选这邻居节点的方式可以改进
        # 获得邻居
        neighbors = angular_neighbors(bvecs, self.num_neb)
        # print('neighbors：', neighbors)
        # 获得邻居和自己的索引，但是元组类型，但是需要自身的，因为后面做GFT变换的时候要做邻接矩阵，只是后面拟合的时候不把自己的信息加入即可
        b_index = [(index,) + tuple(neighbors[index]) for index in range(slice_neb.shape[0])]
        # 转换成数组类型
        b_index_array = np.array(b_index)  # 所有方向的邻接矩阵
        # print('J-neighbors', b_index_array)
        # 当前节点的邻居节点和自己的
        now_neighbors = b_index_array[0]
        # print('now_neighbors', now_neighbors)
        # print('now_neighbors[0]',now_neighbors[0])

        neb_val = bvals[now_neighbors].reshape(self.num_neb + 1, 1)
        neb_vec = bvecs[now_neighbors].reshape(self.num_neb + 1, 3)

        W = get_Gft(neb_val, neb_vec, 'Haar',self.num_neb)
        # 根绝方向上的和预测方向最接近的几个邻居的邻接度
        W = W[0, 1:] # 应该是一个 1 * self.num_neb的张量
        # print('W',W)
        # 这里应该是邻接点个数加1，邻接点个数

        slice_neb = slice_neb[now_neighbors, ...]  # 以第一个节点为预测节点


        bvals = tmp_vec[0]
        bvecs = tmp_vec[1:]
        condition = np.hstack((bvals, bvecs))
        # print('neb_vec',neb_vec.shape)
        return_dict = dict()
        return_dict['lr'] = torch.FloatTensor(slice_neb[0])  # 是对应的，对齐就好了
        return_dict['slice_neb'] = torch.FloatTensor(slice_neb[1:])
        return_dict['index_slice'] = idx_slice
        return_dict['bvals'] = neb_val
        return_dict['bvecs'] = neb_vec
        return_dict['index_grad'] = id_grad
        return_dict['W'] = torch.tensor(W, dtype=torch.float32)
        return_dict['condition'] = condition
        return_dict['sizeof'] = np.array([0.125, 0.125, 0.125])

        return return_dict

    def __len__(self):
        hf = h5py.File(self.folder, 'r')
        # print("hf['{}'.format(self.key)].shape[0] ",hf['{}'.format(self.key)].shape[0])
        return (hf['{}'.format(self.key)].shape[0]) * 6   # 所有训练可用的切片数目
