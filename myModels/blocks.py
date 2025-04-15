import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from myUtils.utilization import get_config

##################################################################################
# GraphConv layers
##################################################################################

class UFGConv(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, shrinkage=None, threshold=1e-4, bias=True,config=None):
        super(UFGConv, self).__init__()
        self.Lev = Lev
        self.num_nodes = num_nodes
        self.shrinkage = shrinkage
        self.threshold = threshold
        self.in_features = in_features
        self.out_features = out_features
        self.crop_len = (Lev - 1) * num_nodes
        self.device= torch.device('cuda:{}'.format(config['gpu_ids'])) if config['gpu_ids'] else torch.device('cpu')
        if torch.cuda.is_available():

            self.weight = nn.Parameter(torch.Tensor(in_features, out_features).to(self.device))
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1).to(self.device))

        else:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1))

        if bias:
            if torch.cuda.is_available():

                self.bias = nn.Parameter(torch.Tensor(out_features).to(self.device))

            else:

                self.bias = nn.Parameter(torch.Tensor(out_features))

        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, d_list):
        # 这里的x是90张图片的数据
        # d_list is a list of matrix operators (torch sparse format), row-by-row
        # x is a torch dense tensor
        # torch.matmul是tensor的乘法，输入可以是高维的。
        # weight的尺寸：num_features, nhid,
        x = x.to(self.device)

        self.weight = self.weight.to(self.device)
        # print(type(x))
        # print('x.max', x.shape)
        x = torch.matmul(x, self.weight).to(self.device)

        # print('x.max', x.max())  # batch neb_num outfeatuers  [2, 89, 11520]

        # Fast Tight Frame Decomposition
        # print('d_list.max', d_list.max())  # [2, 360, 90]
        # the output x has shape [r * Lev * num_nodes, #Features]
        # 46 * 16
        """
        2023.03.08
        这里是用得到的GFT相似系数来初始话权重weight还是像这篇论中这样对数据进行分解重建
        这两种那种情况理论上最优
        感觉对于这种将权重赋给weight就是想当于给网络一个预训练参数，这种能达到的效果可能是不太能改进的
        而将信号分解到不同的滤波段，然后进行卷积，后面再将卷积得到的数据进行重建，这种再不同域上学习的方式不知道可不可以

        另外可不可以将3d数据进行卷积之后再传入到图卷积进行学习然后生成HR
        """

        # Fast Tight Frame Decomposition
        # 但是注意这里在行上连接，是扩展行进行连接，在列上连接是扩展列连接。在dim=0在行上连接
        # print('1:x',x.shape)
        # print('1:d: ', torch.cat(d_list, dim=0))  # 184 * 16
        y = torch.reshape(d_list, (d_list.shape[0], -1, d_list.shape[2]))
        # print('y.shape',y.shape)
        y = y[:, :, 1:]
        # print("y.shape",y.shape)
        # print("x.shape", x.shape)
        x = torch.bmm(y.to(torch.float32).to(self.device), x)
        # x = torch.sparse.mm(torch.cat(d_list, dim=0).to(device), x)
        # print('2:x', x.shape)# 184 * 16
        # the output x has shape [r * Lev * num_nodes, #Features]

        # perform wavelet shrinkage (optional)
        if self.shrinkage is not None:
            if self.shrinkage == 'soft':
                x = torch.mul(torch.sign(x),
                              (((torch.abs(x) - self.threshold) + torch.abs(torch.abs(x) - self.threshold)) / 2))
            elif self.shrinkage == 'hard':
                x = torch.mul(x, (torch.abs(x) > self.threshold))
            else:
                raise Exception('Shrinkage type is invalid')

        # Hadamard product in spectral domain
        self.filter.to(self.device)
        x = self.filter * x
        # print('3:x', x.shape)# 184 * 16
        # filter has shape [r * Lev * num_nodes, 1]
        # the output x has shape [r * Lev * num_nodes, #Features]

        # Fast Tight Frame Reconstruction
        d_list = d_list[:, :, ...]
        print("d_list_2.shape:", d_list.shape)  # [2, 270, 90]
        y = torch.reshape(d_list, (d_list.shape[0], d_list.shape[2], -1))
        # print("y_2.shape:", y.shape)  # [2, 90, 270]
        x = torch.bmm(y.to(torch.float32).to(self.device), x)
        # x = torch.sparse.mm(torch.cat(d_list[:], dim=1).to(device), x[:, :])
        # 不加低通就只能学到边缘信息
        # 这里是不要低通了？
        # x = torch.sparse.mm(torch.cat(d_list[:], dim=1).to(device), x[:, :])
        # 这里的得到的x是默认为所有节点的HR数据
        if self.bias is not None:
            x += self.bias.to(self.device)
        # print('4:x', x.shape)  # 46 * 16
        return x


def rigrsure(x, N1, N2, col_idx):
    """
    Adaptive threshold selection using principle of Stein's Unbiased Risk Estimate (SURE).

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param N1: torch dense tensor with shape [num_nodes, num_hid_features]
    :param N2: torch dense tensor with shape [num_nodes, num_hid_features]
    :param col_idx: torch dense tensor with shape [num_hid_features]
    :return: thresholds stored in a torch dense tensor with shape [num_hid_features]
    """
    n, m = x.shape

    sx, _ = torch.sort(torch.abs(x), dim=0)
    sx2 = sx ** 2
    CS1 = torch.cumsum(sx2, dim=0)
    risks = (N1 + CS1 + N2 * sx2) / n
    best = torch.argmin(risks, dim=0)
    thr = sx[best, col_idx]

    return thr


def multiScales(x, r, Lev, num_nodes):
    """
    计算高频小波系数的尺度，用于小波收缩
    calculate the scales of the high frequency wavelet coefficients, which will be used for wavelet shrinkage.

    :param x: all the blocks of wavelet coefficients, shape [r * Lev * num_nodes, num_hid_features] torch dense tensor
    :param r: an integer
    :param Lev: an integer
    :param num_nodes: an integer which denotes the number of nodes in the graph
    :return: scales stored in a torch dense tensor with shape [(r - 1) * Lev] for wavelet shrinkage
    """
    for block_idx in range(Lev, r * Lev):
        if block_idx == Lev:
            specEnergy_temp = torch.unsqueeze(torch.sum(x[block_idx * num_nodes:(block_idx + 1) * num_nodes, :] ** 2),
                                              dim=0)
            specEnergy = torch.unsqueeze(torch.tensor(1.0), dim=0).to(x.device)
        else:
            specEnergy = torch.cat((specEnergy,
                                    torch.unsqueeze(
                                        torch.sum(x[block_idx * num_nodes:(block_idx + 1) * num_nodes, :] ** 2),
                                        dim=0) / specEnergy_temp))

    assert specEnergy.shape[0] == (r - 1) * Lev, 'something wrong in multiScales'
    return specEnergy


def simpleLambda(x, scale, sigma=1.0):
    """
    De-noising by Soft-thresholding. Author: David L. Donoho

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param scale: the scale of the specific input block of wavelet coefficients, a zero-dimensional torch tensor
    :param sigma: a scalar constant, which denotes the standard deviation of the noise
    :return: thresholds stored in a torch dense tensor with shape [num_hid_features]
    """
    b,n, m = x.shape
    thr = (math.sqrt(2 * math.log(n)) / math.sqrt(n) * sigma) * torch.unsqueeze(scale, dim=0).repeat(m)

    return thr


def waveletShrinkage(x, thr, mode='soft'):
    """
    Perform soft or hard thresholding. The shrinkage is only applied to high frequency blocks.

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param thr: thresholds stored in a torch dense tensor with shape [num_hid_features]
    :param mode: 'soft' or 'hard'. Default: 'soft'
    :return: one block of wavelet coefficients after shrinkage. The shape will not be changed
    """
    assert mode in ('soft', 'hard'), 'shrinkage type is invalid'

    if mode == 'soft':
        x = torch.mul(torch.sign(x), (((torch.abs(x) - thr) + torch.abs(torch.abs(x) - thr)) / 2))
    else:
        x = torch.mul(x, (torch.abs(x) > thr))

    return x

class SUFGConv(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, shrinkage, sigma, bias=True,config=None):
        super(SUFGConv, self).__init__()
        self.r = r
        self.Lev = Lev
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.shrinkage = shrinkage
        self.sigma = sigma
        self.crop_len = (Lev - 1) * num_nodes
        self.device = torch.device('cuda:{}'.format(config['gpu_ids'])) if config['gpu_ids'] else torch.device('cpu')
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features).to(self.device))
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1).to(self.device))

        else:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1))

        if bias:
            if torch.cuda.is_available():
                self.bias = nn.Parameter(torch.Tensor(out_features).to(self.device))
            else:
                self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, d_list):
        # d_list is a list of matrix operators (torch sparse format), row-by-row
        # x is a torch dense tensor
        x = x.to(self.device)
        self.weight = self.weight.to(self.device)
        x = torch.matmul(x, self.weight)
        # print('x.max',x.max()) # batch neb_num outfeatuers  [2, 89, 11520]

        # Fast Tight Frame Decomposition
        # print('d_list.max', d_list.max()) # [2, 360, 90]

        y = torch.reshape(d_list,(d_list.shape[0],-1,d_list.shape[2]))

        y = y[:,:,1:]

        # print("y.max:",y.shape) # [2, 360, 89]
        x = torch.bmm(y.to(torch.float32).to(self.device), x)
        # the output x has shape [r * Lev * num_nodes, #Features]

        # Hadamard product in spectral domain
        x = self.filter * x
        # filter has shape [r * Lev * num_nodes, 1]
        # the output x has shape [r * Lev * num_nodes, #Features]
        # print('x.max', x.max())
        # calculate the scales for thresholding
        ms = multiScales(x, self.r, self.Lev, self.num_nodes)

        # print('wsx_x.shape',x.shape) # [2, 360, 11520]
        # perform wavelet shrinkage
        for block_idx in range(self.Lev - 1, self.r * self.Lev):
            ms_idx = 0
            if block_idx == self.Lev - 1:  # low frequency block
                x_shrink = x[:,block_idx * self.num_nodes:(block_idx + 1) * self.num_nodes, :]
            else:  # remaining high frequency blocks with wavelet shrinkage # 具有小波收缩的剩余高频块
                x_shrink = torch.cat((x_shrink,waveletShrinkage(x[:,block_idx * self.num_nodes:(block_idx + 1) * self.num_nodes, :],
                                                       simpleLambda(x[:,block_idx * self.num_nodes:(block_idx + 1) * self.num_nodes, :],
                                                                    ms[ms_idx], self.sigma), mode=self.shrinkage)), dim=1)
                ms_idx += 1
        # print("x_shrink.shape",x_shrink.shape) # 2, 270, 11520]
        # print('x.max', x_shrink.max())
        # Fast Tight Frame Reconstruction
        # x_shrink = torch.sparse.mm(torch.cat(d_list[(self.Lev - 1)*46:], dim=1).to(device), x_shrink)
        d_list = d_list[:,self.num_nodes:,...]
        # print("d_list_2.shape:",d_list.shape)# [2, 270, 90]
        y = torch.reshape(d_list,(d_list.shape[0],d_list.shape[2],-1))
        # print("y_2.shape:", y.shape) # [2, 90, 270]
        x_shrink = torch.bmm(y.to(torch.float32).to(self.device), x_shrink)
        if self.bias is not None:
            x_shrink += self.bias
        return x_shrink



##################################################################################
# Basic Blocks
##################################################################################
# 残差输入输出维度不一致，做残差要额外处理维度不同的问题，2个卷积层的残差块
class DResBlock(nn.Module):  # extended resblock with different in/ out dimension
    def __init__(self, in_dim, out_dim, kw=4, norm='sn', activation='relu', pad_type='zero', downsample=None, padding=0):
        super(DResBlock, self).__init__()
        self.downsample = downsample
        self.in_dim, self.out_dim = in_dim, out_dim
        self.mid_dim = out_dim
        self.conv1 = Conv2dBlock(self.in_dim, self.mid_dim, kw, 1, norm=norm, activation=activation,
                                 pad_type=pad_type, padding=padding)
        self.conv2 = Conv2dBlock(self.mid_dim, self.out_dim, kw, 1, norm=norm, activation='none', pad_type=pad_type,
                                 padding=padding)
        # 输出输入是否维度一致
        self.learnable_sc = (in_dim != out_dim)
        if self.learnable_sc:
            # 不一致时，不做任何处理只是将数据的维度调整到一致
            self.conv_sc = Conv2dBlock(self.in_dim, self.out_dim, 1, 1, pad_type=pad_type, norm=norm)

    def shortcut(self, x):
        if self.downsample:
            x = self.downsample(x)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return x

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.conv2(h)
        if self.downsample:
            h = self.downsample(h)
        return h + self.shortcut(x)
# 2个残差通道不变的3*3的卷积，残差前后的维度不变，可以直接相加
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out = out + residual
        return out

# num_blocks 个残差块
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

#2D卷积blcok
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            # 自适应性实例化
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            # 生成对抗网络的谱归一化
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x



##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
class NormLayer(nn.Module):
    """Normalization Layers.
    ------------
    # Arguments
        - channels: input channels, for batch norm and instance norm.
        - input_size: input shape without batch size, for layer norm.
    """

    def __init__(self, channels, normalize_shape=None, norm_type='bn'):

        super(NormLayer, self).__init__()

        norm_type = norm_type.lower()
        # initialize normalization
        if norm_type == 'bn':
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == 'gn':
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif norm_type == 'pixel':
            self.norm = lambda x: F.normalize(x, p=2, dim=1)
        elif norm_type == 'ln':
            self.norm = nn.LayerNorm(channels)
        elif norm_type =='adain':
            self.norm = AdaptiveInstanceNorm2d(channels)
        elif norm_type == 'none':
            self.norm = lambda x: x
        else:
            assert 1 == 0, 'Norm type {} not support.'.format(norm_type)

    def forward(self, x):
        # print(x.shape)
        return self.norm(x)

# FAU块
class ReluLayer(nn.Module):
    """Relu Layer.
    ------------
    # Arguments
        - relu type: type of relu layer, candidates are
            - ReLU
            - LeakyReLU: default relu slope 0.2
            - PRelu
            - SELU
            - none: direct pass
    """
    # initialize activation
    def __init__(self, channels, relu_type='relu'):
        super(ReluLayer, self).__init__()
        relu_type = relu_type.lower()

        # initialize activation
        if relu_type == 'relu':
            self.func = nn.ReLU(inplace=True)
        elif relu_type == 'lrelu':
            self.func = nn.LeakyReLU(0.2)
        elif relu_type == 'prelu':
            self.func = nn.PReLU()
        elif relu_type == 'selu':
            self.func = nn.SELU(inplace=True)
        elif relu_type == 'tanh':
            self.func = nn.Tanh()
        elif relu_type == 'leakyrelu':
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == 'none':
            self.func = lambda x: x
        elif relu_type == 'softmax':
            self.func = nn.Softmax(dim=1)
        elif relu_type == 'sigmoid':
            self.func = nn.Sigmoid()
        elif relu_type == 'elu':
            self.func = nn.ELU()
        else:
            assert 1 == 0, 'Relu type {} not support.'.format(relu_type)

    def forward(self, x):

        return self.func(x)


class FAU_ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(FAU_ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResidualBlock(dim, dim,relu_type = activation, norm_type =norm, scale='none',hg_depth=2, att_name='spar',pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale='none', norm_type='none', relu_type='none',pad_type='zero',
                 use_pad=True):
        super(ConvLayer, self).__init__()
        self.use_pad = use_pad
        kernel = 4 if scale == 'down' else 3
        bias = True if norm_type in ['pixel', 'none'] else False
        stride = 2 if scale == 'down' else 1

        kernel = 5 if scale == 'up' else 3


        self.scale_func = lambda x: x
        if scale == 'up':
            self.scale_func = lambda x: nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        # initialize padding
        padding = 2 if scale == 'up' else 1
        if pad_type == 'reflect':
            self.reflection_pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.reflection_pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.reflection_pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        # initialize convolution
        if norm_type == 'sn':
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels,kernel_size=kernel, stride=stride,bias=bias))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels,kernel_size=kernel, stride=stride,bias=bias)

        self.relu = ReluLayer(out_channels, relu_type)
        if norm_type == 'sn':
            norm_type='none'
        self.norm = NormLayer(out_channels, norm_type=norm_type)

    def forward(self, x):
        out = self.scale_func(x)
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        out = self.relu(out)
        return out

class ResidualBlock(nn.Module):
    """
    Residual block recommended in: http://torch.ch/blog/2016/02/04/resnets.html
    ------------------
    # Args
        - hg_depth: depth of HourGlassBlock. 0: don't use attention map.
        - use_pmask: whether use previous mask as HourGlassBlock input.
    """
    def __init__(self, c_in, c_out ,relu_type='prelu', norm_type='bn',scale='none', hg_depth=3, att_name='spar',pad_type='zero'):
        super(ResidualBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out

        self.norm_type = norm_type

        self.relu_type = relu_type

        self.hg_depth = hg_depth

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        # 下采样层
        if scale == 'none' and c_in == c_out:
            self.shortcut_func = lambda x: x
        else:
            self.shortcut_func = ConvLayer(c_in, c_out,scale,pad_type=pad_type)


        if scale == 'down':
            scales = ['none', 'down']
        elif scale == 'up':
            scales = ['up', 'none']
        elif scale == 'none':
            scales = ['none', 'none']

        self.conv1 = ConvLayer(c_in , c_out, scales[0] , pad_type = pad_type, **kwargs)
        self.conv2 = ConvLayer(c_out , c_out, scales[1], pad_type=pad_type, **kwargs)

        if self.norm_type == 'sn':
            self.norm_type = 'none'
        # 规范化
        self.preact_func = nn.Sequential(
            NormLayer(c_in, norm_type=self.norm_type),
            ReluLayer(c_in, self.relu_type),
        )

        if att_name.lower() == 'spar':
            c_attn = 1
        elif att_name.lower() == 'spar3d':
            c_attn = c_out
        else:
            raise Exception("Attention type {} not implemented".format(att_name))

        self.att_func = HourGlassBlock(self.hg_depth, c_out, c_attn,pad_type=pad_type, **kwargs)

    def forward(self, x):

        identity = self.shortcut_func(x)

        out = self.preact_func(x)

        out = self.conv1(out)


        out = self.conv2(out)

        # 最后的残差
        out = identity + self.att_func(out)
        return out


class HourGlassBlock(nn.Module):
    """Simplified HourGlass block.
    Reference: https://github.com/1adrianb/face-alignment
    --------------------------
    """

    def __init__(self, depth, c_in, c_out,
                 c_mid=64,
                 norm_type='bn',
                 relu_type='prelu',
                 pad_type='zero'
                 ):
        super(HourGlassBlock, self).__init__()
        self.depth = depth
        self.c_in = c_in
        # 卷积核个数？
        self.c_mid = c_mid
        self.pad_type = pad_type
        self.c_out = c_out

        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if self.depth:
            self._generate_network(self.depth)
            self.out_block = nn.Sequential(
                ConvLayer(self.c_mid, self.c_out,norm_type='none', relu_type='none',pad_type= self.pad_type),
                nn.Sigmoid()
            )

    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs))
        self.add_module('b2_' + str(level), ConvLayer(c1, c2,scale='down', **self.kwargs))
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid , pad_type= self.pad_type, **self.kwargs))

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', pad_type= self.pad_type,**self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])

        return up1 + up2

    def forward(self, x, pmask=None):
        if self.depth == 0: return x
        input_x = x

        x = self._forward(self.depth, x)
        # 得到注意力图
        self.att_map = self.out_block(x)
        x = input_x * self.att_map
        return x
