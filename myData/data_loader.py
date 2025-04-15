from torch.utils.data.dataset import Dataset
import numpy as np
import h5py
import torch
import torch.utils.data
from myUtils.torchutils import get_uniform_downsample_indices
from myUtils.utilization import angular_neighbors,get_Gft



class getTrainData(Dataset):
    '''
    Dataloader for the proposed framework
    '''
    def __init__(self, folder, mode='train',num_grad=90,num_slice=72,pad_size=(72,80),config=None):
        self.mode = mode
        self.config = config
        self.pad_size = pad_size
        self.grad_able = 20
        self.num_slice = num_slice
        self.num_grad = num_grad
        self.num_neb =config['num_neb']
        if self.mode=='train':
            self.key = 'train'
        elif self.mode=='test':
            print("the main of getTrainData's method can't be applied to Test!")
        elif self.mode == 'val':
            print("the main of getTrainData's method can't be applied to Val!")

        self.folder = folder
        self.dataset = 'HCP'

        print(self.dataset, 'numbel of num_grad: {}'.format(self.num_grad))
        hf = h5py.File(self.folder, 'r')
        self.bval_max = (hf['{}bvec'.format(self.key)][:,: ,0]).max()
        hf.close()
        print(self.dataset, 'bval max: {}'.format(self.bval_max))

    def __getitem__(self, item):
        ran = torch.randint(0, self.__len__(), ()).numpy()
        idx_slice,id_grad = ran // (self.grad_able) ,ran % (self.grad_able)  # the order of slice and grad

        hf = h5py.File(self.folder, 'r')

        slice_all = hf['{}'.format(self.key)][idx_slice]  # get the data from dataset


        cond_all = hf['{}bvec'.format(self.key)][idx_slice]
        cond_list = get_uniform_downsample_indices(cond_all[:, 1:], 30,0) # 输出选择的点在原始点集中的索引,第一次下采样的索引，用于自监督
        neb_vec = get_uniform_downsample_indices(cond_all[cond_list, 1:], 10,0) # 输出选择的点在原始点集中的索引，第二次下采样，用于训练
        neb_vec = [cond_list[val] for i,val in enumerate(neb_vec)] # 取出这10个点在cond_all中的索引




        slice_train = slice_all[cond_list]  # 挑选出构建训练集的数据
        # q空间两次下采样均匀采样
        slice_neb = slice_all[neb_vec] # 训练集中的邻居节点

        pre_list = list(set(cond_list).difference(neb_vec)) # 需要预测的索引

        self.pre_id = pre_list[id_grad]
        slice_pre = slice_all[self.pre_id]  # 训练时的预测节点

        # 裁剪图像
        slice_neb = slice_neb[:, 10:self.pad_size[0] + 10, 20:self.pad_size[1] + 20]
        slice_pre = slice_pre[10:self.pad_size[0] + 10, 20:self.pad_size[1] + 20]

        tmp = np.expand_dims(slice_pre, axis=0)
        slice_neb = np.row_stack((tmp, slice_neb))
        cond_list = cond_all[cond_list]



        tmp_vec = cond_all[self.pre_id]

        neb_vec = cond_all[neb_vec]


        neb_vec = np.row_stack((tmp_vec, neb_vec))


        index_slice = (64 + (idx_slice % self.num_slice)) * 0.01



        bvals = neb_vec[:, 0] * 1.0 / self.bval_max  # b值归一化
        bvecs = neb_vec[:, 1:]


        neighbors = angular_neighbors(bvecs, self.num_neb)

        b_index = [(index,) + tuple(neighbors[index]) for index in range(slice_neb.shape[0])]

        b_index_array = np.array(b_index)  # 所有方向的邻接矩阵

        now_neighbors = b_index_array[0]


        neb_val = bvals[now_neighbors].reshape(self.num_neb + 1, 1)
        neb_vec = bvecs[now_neighbors].reshape(self.num_neb + 1, 3)

        W = get_Gft(neb_val, neb_vec, 'Haar',self.num_neb)

        W = W[0, 1:]


        slice_neb = slice_neb[now_neighbors, ...]


        bvals = tmp_vec[0]
        bvecs = tmp_vec[1:]
        condition = np.hstack((bvals, bvecs))
        return_dict = dict()
        return_dict['lr'] = torch.FloatTensor(slice_neb[0])
        return_dict['slice_neb'] = torch.FloatTensor(slice_neb[1:])
        return_dict['index_slice'] = index_slice
        return_dict['bvals'] = neb_val
        return_dict['bvecs'] = neb_vec
        return_dict['W'] = torch.tensor(W, dtype=torch.float32)
        return_dict['condition'] = condition
        return_dict['sizeof'] = np.array([0.125, 0.125, 0.125])

        return return_dict

    def __len__(self):
        hf = h5py.File(self.folder, 'r')

        return (hf['{}'.format(self.key)].shape[0]) * (self.grad_able)    # 所有训练可用的切片数目


class getTestANDValData(Dataset):
    '''
    Dataloader for the proposed framework
    '''

    def __init__(self, folder, mode='test', num_grad=90, num_slice=72, pad_size=(144, 160), config=None):
        self.mode = mode
        print("mode",mode)
        self.config = config
        self.pad_size = pad_size
        self.grad_able = config['able_grad']
        self.ran = 0
        self.num_slice = num_slice
        self.num_grad = num_grad
        self.num_neb = config['num_neb']
        self.test_grad = 84
        if self.mode == 'train':
           print('Error!This is the only way to load test data')
        elif self.mode == 'test':
            self.key = 'test'
        elif self.mode == 'val':
            self.key = 'val'
        self.folder = folder

        self.dataset = 'HCP'
        print(self.dataset, 'In Testdataset the numbel of num_grad: {}'.format(self.num_grad))

        hf = h5py.File(self.folder, 'r')
        # 最大b值
        self.bval_max = (hf['{}bvec'.format(self.key)][:, :, 0]).max()
        hf.close()
        print(self.dataset, 'bval max: {}'.format(self.bval_max))
    def __getitem__(self, item):

        idx_slice,id_grad = self.ran // (self.test_grad ), self.ran % (self.test_grad)
        self.ran = self.ran+1
        hf = h5py.File(self.folder, 'r')
        # print('ran',self.ran)
        slice_all = hf['{}'.format(self.key)][idx_slice]
        cond_all = hf['{}bvec'.format(self.key)][idx_slice]
        cond_list = get_uniform_downsample_indices(cond_all[:, 1:], 6,0)
        slice_train = slice_all[cond_list]
        all_index = [i for i in range(0,90)]
        pre_list = list(set(all_index).difference(cond_list))

        self.pre_id = pre_list[id_grad]

        slice_pre = slice_all[self.pre_id]



        tmp = np.expand_dims(slice_pre, axis=0)
        slice_neb = np.row_stack((tmp, slice_train))


        neb_vec = cond_all[cond_list]

        tmp_vec = cond_all[self.pre_id]



        neb_vec = np.row_stack((tmp_vec, neb_vec))
        index_slice = (64 + (idx_slice % self.num_slice)) * 0.01


        bvals = neb_vec[:, 0] * 1.0 / self.bval_max  # b值归一化
        bvecs = neb_vec[:, 1:]


        neighbors = angular_neighbors(bvecs, self.num_neb)

        b_index = [(index,) + tuple(neighbors[index]) for index in range(slice_neb.shape[0])]

        b_index_array = np.array(b_index)

        now_neighbors = b_index_array[0]


        neb_val = bvals[now_neighbors].reshape(self.num_neb + 1, 1)
        neb_vec = bvecs[now_neighbors].reshape(self.num_neb + 1, 3)

        W = get_Gft(neb_val, neb_vec, 'Haar',self.num_neb)

        W = W[0, 1:]


        slice_neb = slice_neb[now_neighbors, ...]


        bval = tmp_vec[0]
        bvec = tmp_vec[1:]
        condition = np.hstack((bval, bvec))

        return_dict = dict()
        return_dict['lr'] = torch.FloatTensor(slice_neb[0])
        return_dict['slice_neb'] = torch.FloatTensor(slice_neb[1:])
        return_dict['index_slice'] = index_slice
        return_dict['bvals'] = neb_val
        return_dict['bvecs'] = neb_vec
        return_dict['W'] = torch.tensor(W, dtype=torch.float32)
        return_dict['condition'] = condition
        return_dict['sizeof'] = np.array([0.125, 0.125, 0.125])
        return return_dict


    def __len__(self):
        hf = h5py.File(self.folder, 'r')
        if self.key=='test':
            return (hf['{}'.format(self.key)].shape[0])*  (self.test_grad )
        else:
            return (hf['{}'.format(self.key)].shape[0]//7) * (self.test_grad)