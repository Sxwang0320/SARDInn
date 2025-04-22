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



# HCP测试集的dataloader函数
class getTrainData(Dataset):
    '''
    Dataloader for the proposed framework with q-space augmentation
    '''
    def __init__(self, folder, mode='train',num_grad=90,grad_able=30,num_slice=72,pad_size=(72,80),config=None):
        self.mode = mode
        self.config = config
        self.pad_size = pad_size
        self.grad_able = grad_able        # 可用bvec数码
        self.num_slice = num_slice # 每个bvec的DWI数
        self.num_grad = num_grad   # 全部bvec数目
        self.num_neb = 10          # 邻域方向数目
        self.key = self.mode

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
        # self.num_grad-self.grad_able 表示需要重建bvec的数量，
        idx_slice,id_grad = self.ran // (self.num_grad-self.grad_able), self.ran % (self.num_grad-self.grad_able)  # the order of slice and grad
        self.ran = self.ran + 1  # 更新ran
        hf = h5py.File(self.folder, 'r')

        slice_all = hf['{}'.format(self.key)][idx_slice]  # get the data from dataset

        # 当前实例的所有方向
        cond_all = hf['{}bvec'.format(self.key)][idx_slice]
        cond_list = get_uniform_downsample_indices(cond_all[:, 1:], self.grad_able,0) # 找出可用的bvec索引

        # q空间两次下采样均匀采样
        slice_neb = slice_all[cond_list] # 训练集中的邻居节点
        all_index = [i for i in range(0, 90)] # 总的bvec数目
        pre_list = list(set(all_index).difference(cond_list))  # 需要重建的bvec索引

        self.pre_id = pre_list[id_grad]

        slice_pre = slice_all[self.pre_id]  # 重建目标DWI

        # 裁剪图像

        # slice_neb = slice_neb[:, 10:self.pad_size[0] + 10, 20:self.pad_size[1] + 20]
        # slice_pre = slice_pre[10:self.pad_size[0] + 10, 20:self.pad_size[1] + 20]

        tmp = np.expand_dims(slice_pre, axis=0) # 扩展维度
        slice_neb = np.row_stack((tmp, slice_neb))  # 姜预测的方向在扩展的维度上加上已知方向

        tmp_vec = cond_all[self.pre_id] # 预测方向

        neb_vec = cond_all[cond_list] # 邻居方向


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
        print("hf['{}'.format(self.key)].shape[0] ",hf['{}'.format(self.key)].shape[0])
        return (hf['{}'.format(self.key)].shape[0]) * (self.num_grad - self.grad_able)   # 所有训练可用的切片数目


# 直接预测的dataloader函数
class getValData(Dataset):
    '''
    Dataloader for the proposed framework with q-space augmentation
    '''
    def __init__(self, folder, mode='train',num_grad=90,grad_able=30,num_slice=72,pad_size=(72,80),config=None):
        self.mode = mode
        self.config = config
        self.pad_size = pad_size
        self.grad_able = grad_able        # 可用bvec数码
        self.num_slice = num_slice # 每个bvec的DWI数
        self.num_grad = num_grad   # 全部bvec数目
        self.num_neb = 10          # 邻域方向数目
        self.key = self.mode

        self.folder = folder
        self.ran=0
        self.dataset = 'HCP'

        print(self.dataset, 'numbel of num_grad: {}'.format(self.num_grad))
        hf = h5py.File(self.folder, 'r')
        # 最大b值
        self.bval_max = (hf['{}re_bvec'.format(self.mode)][:,: ,0]).max()
        hf.close()
        print(self.dataset, 'bval max: {}'.format(self.bval_max))

    def __getitem__(self, item):
        # self.num_grad-self.grad_able 表示需要重建bvec的数量，
        idx_slice,id_grad = self.ran // (self.num_grad-self.grad_able), self.ran % (self.num_grad-self.grad_able)  # the order of slice and grad
        self.ran = self.ran + 1  # 更新ran
        hf = h5py.File(self.folder, 'r')

        slice_neb = hf['{}able_dwi'.format(self.key)][idx_slice]  # 可用梯度dwi
        slice_pre = hf['{}re_dwi'.format(self.key)][idx_slice,id_grad] # 预测bvec的DWI



        tmp = np.expand_dims(slice_pre, axis=0) # 扩展维度
        slice_neb = np.row_stack((tmp, slice_neb))  # 姜预测的方向在扩展的维度上加上已知方向

        tmp_vec = hf['{}re_bvec'.format(self.key)][idx_slice,id_grad] # 预测bvec

        neb_vec = hf['{}able_bvec'.format(self.key)][idx_slice] # 邻居方向


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
        print("hf['{}re_bvec'.format(self.key)].shape[0] ",hf['{}re_bvec'.format(self.key)].shape[0]* (self.num_grad - self.grad_able))
        return (hf['{}re_bvec'.format(self.key)].shape[0]) * (self.num_grad - self.grad_able)   # 所有训练可用的切片数目
