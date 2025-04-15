import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)

from myModels.rdn import RDB
# -------------------------------
# SRResNet
# <Ledig, Christian, et al. "Photo-realistic single image super-resolution
# using a generative adversarial network.">
# -------------------------------
def RepeatBVector( bvec, spatial_dims):
    multiples = [1,1,1] + list(spatial_dims)
    for _ in spatial_dims:
        bvec = bvec.unsqueeze(-1)
    bvec = torch.tile(bvec,multiples)
    return bvec

def conv(ni, nf, kernel_size=3, actn=False):
    layers = [nn.Conv2d(ni, nf, kernel_size, padding='same')]
    if actn: layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)

class ResSequential(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.m = nn.Sequential(*layers)

    def forward(self, x): return x + self.m(x) * self.res_scale

def res_block(nf):
    return ResSequential(
        [conv(nf, nf, actn=True), conv(nf, nf)],
        1.0)  # this is best one

class SRResnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nf = args['G0']
        self.feature_dim = args['laten_dim']
        features = [conv(1, self.nf)]
        for i in range(18): features.append(res_block(self.nf))
        features += [conv(self.nf, self.nf),
                     conv(self.nf, self.feature_dim)]
        self.features = nn.Sequential(*features)
        self.conv_jiaqun = nn.Sequential(
            *[nn.Conv2d(in_channels=self.feature_dim * 9, out_channels=self.feature_dim, kernel_size=(1, 1),
                        padding='same'),
              nn.ReLU(),
              # nn.Conv2d(in_channels=288, out_channels=32, kernel_size=(1, 1), padding='same'),
              # nn.ReLU(),
              # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding='same'),
              # nn.ReLU(),
              ])
    def forward(self, x):
        # print("x.shape,",x.shape)
        x = x.reshape(-1, 1, x.shape[1], x.shape[2])
        # print("x.shape,", x.shape)
        return self.features(x)



# -------------------------------
# ResCNN encoder network
# <Du, Jinglong, et al. "Super-resolution reconstruction of single
# anisotropic 3D MR images using residual convolutional neural network.">
# -------------------------------
class ResCNN(nn.Module):
    def __init__(self, args):
        super(ResCNN, self).__init__()
        self.input_dim = args['input_dim']
        self.laten_dim = args['laten_dim']
        self.conv_start = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=3 // 2),
            nn.ReLU(inplace=True)
        )
        self.conv_end = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels= self.laten_dim, kernel_size=3, padding=3 // 2),
        )
        self.conv_jiaqun = nn.Sequential(
            *[nn.Conv2d(in_channels=self.laten_dim * 9, out_channels=self.laten_dim, kernel_size=(1, 1), padding='same'),
              nn.ReLU(),
              # nn.Conv2d(in_channels=288, out_channels=32, kernel_size=(1, 1), padding='same'),
              # nn.ReLU(),
              # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding='same'),
              # nn.ReLU(),
              ])

    def forward(self, x):
        # print("x.shape,",x.shape)
        x = x.reshape(-1, 1, x.shape[1], x.shape[2])
        # print("x.shape,", x.shape)
        in_block1 = self.conv_start(x)
        out_block1 = self.block1(in_block1)
        in_block2 = out_block1 + in_block1
        out_block2 = self.block2(in_block2)
        in_block3 = out_block2 + in_block2
        out_block3 = self.block3(in_block3)
        res_img = self.conv_end(out_block3 + in_block3)
        return x + res_img

# -------------------------------
# RDN encoder network
# <Zhang, Yulun, et al. "Residual dense network for image super-resolution.">
# Here code is modified from: https://github.com/yjn870/RDN-pytorch/blob/master/models.py
# -------------------------------
class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        self.args = args

        self.G0 = args['G0']
        self.kSize = args['RDNkSize']

        self.n_colors =  args['input_dim']
        self.outdim=args['laten_dim']
        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
            'C': (8, 3, 64),
        }[args['RDNconfig']]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(self.n_colors, self.G0, self.kSize, padding='same', stride=1)
        self.SFENet2 = nn.Conv2d(self.G0, self.G0, self.kSize,padding='same', stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList([RDB(growRate0 = self.G0, growRate = G, nConvLayers = C)])
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G, self.G0, 1, padding=0, stride=1),
            nn.Conv2d(self.G0, self.G0, self.kSize, padding='same', stride=1)
        ])
        # 10.19准备做这个加权的消融实验
        self.conv_jiaqun =nn.Sequential(*[nn.Conv2d(in_channels=self.outdim*9, out_channels=self.outdim, kernel_size=(1, 1), padding='same'),
                                          nn.ReLU(),
                                          # nn.Conv2d(in_channels=288, out_channels=32, kernel_size=(1, 1), padding='same'),
                                          # nn.ReLU(),
                                          # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding='same'),
                                          # nn.ReLU(),
                                          ])

        self.output=nn.Conv2d(self.G0,self.outdim,kernel_size=self.kSize,padding='same')

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[1], x.shape[2])
        f__1 = self.SFENet1(x) # torch.Size([30, 32, 120, 140])
        x = self.SFENet2(f__1) # torch.Size([30, 32, 120, 140])

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1
        out = self.output(x)
        return out



# -------------------------------
# decoder implemented by a simple MLP
# -------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_1(nn.Module):
    def __init__(self, input_dim, output_dim,params):
        super(MLP_1, self).__init__()
        depth  = params['MLP_depth']
        width = params['MLP_width']
        in_dim = params['MLP_in_dim']
        out_dim = params['out_dim']  # MLP的输出，也就是得到的隐式神经函数的输出
        # 获取显卡ID
        gpu_ids = params['gpu_ids']
        self.device = torch.device('cuda:{}'.format(gpu_ids)) if gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        stage_one = []
        stage_two = []
        self.activation = nn.ReLU()

        self.layer1=nn.Sequential(nn.Linear(in_dim, width),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(width + 3, width),nn.ReLU())
        self.layer3 =nn.Sequential(nn.Linear(width + 3, width),nn.ReLU())
        self.layer4 = nn.Linear(width + 3, in_dim-3)

        # self.layer5 = nn.Sequential(nn.Conv2d(in_dim, width, kernel_size=1, stride=1),nn.ReLU())
        # self.layer6 = nn.Sequential(nn.Conv2d(width, width, kernel_size=1, stride=1),nn.ReLU())
        # self.layer7 = nn.Sequential(nn.Conv2d(width, width, kernel_size=1, stride=1),nn.ReLU())
        # self.layer8 = nn.Sequential(nn.Conv2d(width, out_dim, kernel_size=1, stride=1),nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(in_dim, width),nn.ReLU())
        self.layer6 = nn.Sequential(nn.Linear(width+3, width),nn.ReLU())
        self.layer7 = nn.Sequential(nn.Linear(width+3, width),nn.ReLU())
        self.layer8 = nn.Sequential(nn.Linear(width+3, out_dim),nn.ReLU())
        # self.fusion=nn.Linear(in_features=6, out_features=1)
    def forward(self, x,bvec_):

        x1 = self.layer1(x.to(self.device)) # torch.Size([2, 256, 120, 140])
        x1_ = torch.cat([x1,bvec_],dim=-1)
        x2 = self.layer2(x1_)
        x2_ = torch.cat([x2, bvec_], dim=-1)
        x3 = self.layer3(x2_)
        x3_ = torch.cat([x3, bvec_], dim=-1)
        h = torch.sin(self.layer4(x3_))
        h_ = h+x[...,:-3]
        h_ = torch.cat([h_, bvec_], dim=-1)
        h1 = self.layer5(h_)
        h1_ = torch.cat([h1, bvec_], dim=-1)
        h2 = self.layer6(h1_)
        h2_ = torch.cat([h2, bvec_], dim=-1)
        h3 = self.layer7(h2_)
        h3_ = torch.cat([h3, bvec_], dim=-1)
        h4 = self.layer8(h3_)
        return h4



