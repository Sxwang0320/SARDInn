import torch.nn.init as init
import math
import torch
from scipy.spatial import distance_matrix
import pulp

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def adjust_learning_rate(optimer, epoch, config):
    lr = config['lr']
    if config['lr_policy']=='cos':
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / config['n_epoch']))
    elif config['lr_policy']=='step':
        for milestone in config['step_size']:
            lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimer.param_groups:
            param_group['lr'] = lr

def get_uniform_downsample_indices(points, num_samples,start_is=None):

    if start_is!=None:
        selected_index=start_is
    else:
        # 随机选择第一个点
        selected_index= torch.randint(len(points), (1,)).item()
    selected_indices = [selected_index]
    remaining_indices = torch.arange(len(points))
    remaining_indices = remaining_indices[remaining_indices != selected_index]
    for _ in range(num_samples - 1):
        # 计算每个剩余点与已选择点之间的球面距离
        distances = distance_matrix(points[remaining_indices], points[selected_indices])
        # 找到与已选择点的平均角距离最大的点
        avg_dist = torch.tensor(distances.mean(axis=1))
        max_dist_index = torch.argmax(avg_dist)
        # 获取其在剩余点集中的索引
        selected_index = remaining_indices[max_dist_index].item()
        # 将其添加到已选择点集中
        selected_indices.append(selected_index)
        # 移除已选择的点
        remaining_indices = remaining_indices[remaining_indices != selected_index]

    return selected_indices




