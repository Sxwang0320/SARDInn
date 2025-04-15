import torch
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from myModels.networks import RDN,ResCNN,SRResnet,MLP_1
from myUtils.torchutils import weights_init
from piqa import PSNR ,SSIM

####################################
# Proposed: SARDInn to DWIs
###################################
class dwi_SARDInn(nn.Module):
    def __init__(self,config):
        super(dwi_SARDInn, self).__init__()

        # set learning rate
        lr = config['lr']
        # Initiate the networkse
        # self.autoEncoder = ResCNN(args=config)
        self.autoEncoder =SRResnet(args=config)
        # self.autoEncoder = Encoder_2D(args=config)
        # self.autoEncoder = RDN(args=config)
        self.myLiif = MLP_1(input_dim=config['input_dim'], output_dim=config['output_dim'],
                                       params=config)
        # Setup the optimizers
        beta1 = config['beta1']
        beta2 = config['beta2']

        # 模型参数总数
        all =  sum([np.prod(p.size()) for p in self.myLiif.parameters()])

        # 需要学习的参数总数
        liif_params = list(self.myLiif.parameters())

        trainable = sum([np.prod(p.size()) for p in liif_params if p.requires_grad])

        print('LIIF Trainable: {}/{}M'.format(trainable / 1000000, all / 1000000))

        encoder_params = list(self.autoEncoder.parameters())

        trainable = sum([np.prod(p.size()) for p in encoder_params if p.requires_grad])

        print('AutoEncoder Trainable: {}/{}M'.format(trainable / 1000000, all / 1000000))

        # 设置优化器
        self.latent_dim = config['latent_dim']
        self.ae_opt = torch.optim.Adam([ {'params': self.autoEncoder.parameters()},
            {'params': self.myLiif.parameters()}], lr=lr,betas=(beta1, beta2),weight_decay=config['weight_decay'])

        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.ae_opt, mode='min', factor=0.1, patience=10,
        #                                                  threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
        #                                                  eps=1e-08, verbose=False)
        # 获取显卡ID
        gpu_ids = config['gpu_ids']
        self.device = torch.device('cuda:{}'.format(gpu_ids)) if gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        print('Deploy to {}'.format(self.device))
        self.autoEncoder.to(self.device)
        self.myLiif.to(self.device)
        torch.cuda.set_device(self.device)# 设置显卡号
        # Network weight initialization
        self.myLiif.apply(weights_init(config['init']))
        self.autoEncoder.apply(weights_init(config['init']))
        print('Init Netwwork with {}'.format(config['init']))

        # 评估函数
        self.criterion = nn.MSELoss()
        self.sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.getPsnr = PSNR()
        self.getPsnr.to(self.device)
        self.getSSIM = SSIM(n_channels=1)
        self.getSSIM.to(self.device)
        self.kl = nn.KLDivLoss(reduction='batchmean')
    # 计算像素级别的损失，L1损失
    def recon_criterion(self, input, target, mse=False):
        # 如果target是0，即为黑则改为1，不是0则为零，来搞成mask，这个mask去掉了所有的零灰度区域
        # brain_mask = 1. - (target  < 3.057).float()
        if mse:
            pix_loss = F.mse_loss(input, target, reduction='mean')
        else:
            # 绝对差值总和平均
            pix_loss =  F.smooth_l1_loss(input, target, reduction='mean')
        # print(brain_mask.sum())
        # pix_loss = pix_loss / (brain_mask.sum() + 1e-10)
        return pix_loss
    # 结合b值的潜在特征进行解码得到重建目标
    def forward(self,lr, index_slice,bvals,bvecs,sizeof,W,slice_neb):

        self.ae_opt.zero_grad()
        W = W.to(self.device)
        bvecs = bvecs.to(self.device)
        slice_neb_ = slice_neb.reshape(slice_neb.shape[0]*slice_neb.shape[1],slice_neb.shape[2],slice_neb.shape[3])
        # sizeof = sizeof.reshape(sizeof.shape[0]*sizeof.shape[1],sizeof.shape[2])
        # print("sizeof.shape", sizeof.shape)
        content_neb = self.autoEncoder.forward(slice_neb_.to(self.device))  # 提取邻居节点的的语义信息
        # print("content_neb.shape", content_neb.shape)  # [16, 14, 120, 140]
        content_neb = content_neb.reshape(slice_neb_.shape[0],self.latent_dim, lr.shape[-2],lr.shape[-1])
        # print("content_neb.shape",content_neb.shape) # [16, 14, 120, 140]

        # 10，19 做消融实验，关于x空间的特征聚合的消融
        feat = F.unfold(content_neb, 3, padding=1)
        # print('feat.shape', feat.shape)# [16, 126, 16800]
        feat = feat.view(content_neb.shape[0], content_neb.shape[1]* 9,content_neb.shape[2],content_neb.shape[3]) # torch.Size([30, 288, 120, 140])
        # print('feat4.shape', feat.shape)
        feat = self.autoEncoder.conv_jiaqun(feat) # torch.Size([30, 128, 120, 140])
        # print('feat4.shape', feat.shape)
        # feat = content_neb
        feat = feat.reshape(slice_neb.shape[0], slice_neb.shape[1], -1, content_neb.shape[2] * content_neb.shape[3]).permute(0,1,3,2).to(self.device)
        # print('feat4.shape', feat.shape)
        preds = []
        areas = []
        bvec = bvecs[:,0]
        for it in range(slice_neb.shape[1]):
            vec = bvecs[:, it + 1]
            bvec_ = (bvec - vec)
            # print("bvec:",bvec.shape) # [32, 4]
            q_feat = feat[:, it]
            # q_feat = rearrange(q_feat,"b c h ->b h c")
            # print("q_feat.shape", q_feat.shape)  # [9, 12000, 72]
            bvec_ = bvec_.reshape(bvec_.shape[0], 1, bvec_.shape[1]).repeat(1, q_feat.shape[1], 1)
            inp = torch.cat([q_feat, bvec_], dim=-1)
            area = W[:, it].reshape(W.shape[0], 1).repeat(1, q_feat.shape[1])
            # area = torch.abs(bvec_[:, :,0] * bvec_[:,:, 1] * bvec_[:,:, 2])
            # print("area",area.shape)
            # print("inp.shape", inp.shape)# [9, 12000, 75]
            pred = self.myLiif(inp.view(inp.shape[0], inp.shape[1], -1), bvec_).view(inp.shape[0], inp.shape[1], -1)
            # print("pred", pred.shape)
            preds.append(pred)
            areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        # print("tot_area", tot_area.shape) # [9, 12000]
        ret = 0
        # print("preds", type(preds))

        # print("pre", pre.shape)
        # 下面是GFT系数加权
        for pred, area in zip(preds, areas):
            # print("pred.shape", pred.shape)
            ret = ret + pred * ((area / tot_area)).unsqueeze(-1)

        # print("loss_all", loss_all)
        # print("ret.shape", ret.shape)  # [9, 12000, 1]
        # print("label.max", lr.max())
        pre = ret
        pre = pre.reshape(lr.shape)

        lr= lr.to('cpu')
        pre = pre.to('cpu')

        pre = pre.reshape(lr.shape[0], lr.shape[1], lr.shape[2])
        """
        MRI图中的数值不仅仅表示灰度，还是带有其他纤维方向的信息，这里试一试不做归一化做损失
        2023.06.28
        """
        # print("pre.shape", pre.max())
        # print("lr.shape", lr.max())

        pre = pre * 5000.
        lr = lr * 5000.
        pre_ = (((pre - pre.min()) * 1.0) / ((pre.max() - pre.min()) * 1.0 + 1e-8)) * 1.0
        lr_ = (((lr - lr.min()) * 1.0) / ((lr.max() - lr.min()) * 1.0 + 1e-8)) * 1.0
        self.loss_l1 = self.recon_criterion(pre.to(self.device),
                                             lr.to(self.device))  # 约束训练MLP网络学习隐式神经函数
        self.loss_l2 = self.criterion(pre.to(self.device),
                                            lr.to(self.device))  # 约束训练MLP网络学习隐式神经函数


        self.psnr = self.getPsnr(pre_.to(self.device), lr_.to(self.device)).item()

        self.SSIM = self.getSSIM(pre_.reshape(lr.shape[0],1, lr.shape[1], lr.shape[2]).to(self.device), lr_.reshape(lr.shape[0],1, lr.shape[1], lr.shape[2]).to(self.device))


        total_loss = self.loss_l1
        # total_loss = self.loss_l2
        total_loss.backward()
        self.ae_opt.step()

        return_dict = {}
        return_dict['pre'] = pre[0].detach().cpu().numpy()
        return_dict['target'] = lr[0].detach().cpu().numpy()
        return_dict['error'] = (lr[0]-pre[0]).detach().cpu().numpy()

        return return_dict, total_loss.item(), self.psnr, self.SSIM.item()


    def sample(self,lr, index_slice,bvals,bvecs,sizeof,W,slice_neb,condition):
        with torch.no_grad():
            self.autoEncoder.eval()
            self.myLiif.eval()
            W = W.to(self.device)
            bvecs = bvecs.to(self.device)
            slice_neb_ = slice_neb.reshape(slice_neb.shape[0] * slice_neb.shape[1], slice_neb.shape[2],
                                           slice_neb.shape[3])
            content_neb = self.autoEncoder.forward(slice_neb_.to(self.device))  # 提取邻居节点的的语义信息
            # print("content_neb.shape", content_neb.shape)  # [16, 14, 120, 140]
            content_neb = content_neb.reshape(slice_neb_.shape[0], self.latent_dim, lr.shape[-2], lr.shape[-1])
            # print("content_neb.shape",content_neb.shape) # [16, 14, 120, 140]


            # 10.19 x 空间聚合的消融实验
            feat = F.unfold(content_neb, 3, padding=1)
            # print('feat.shape', feat.shape)# [16, 126, 16800]
            feat = feat.view(content_neb.shape[0], content_neb.shape[1] * 9, content_neb.shape[2],
                             content_neb.shape[3])  # torch.Size([30, 288, 120, 140])

            feat = self.autoEncoder.conv_jiaqun(feat)  # torch.Size([30, 128, 120, 140])

            # feat = content_neb
            feat = feat.reshape(slice_neb.shape[0], slice_neb.shape[1], -1,
                                content_neb.shape[2] * content_neb.shape[3]).permute(0, 1, 3, 2).to(self.device)
            # print('feat4.shape', feat.shape)
            preds = []
            areas = []
            bvec = bvecs[:, 0]
            for it in range(slice_neb.shape[1]):
                vec = bvecs[:, it + 1]
                bvec_ = (bvec - vec)
                # print("bvec:",bvec.shape) # [32, 4]
                q_feat = feat[:, it]
                # q_feat = rearrange(q_feat,"b c h ->b h c")
                # print("q_feat.shape", q_feat.shape)  # [9, 12000, 72]
                bvec_ = bvec_.reshape(bvec_.shape[0], 1, bvec_.shape[1]).repeat(1, q_feat.shape[1], 1)
                inp = torch.cat([q_feat, bvec_], dim=-1)
                area = W[:, it].reshape(W.shape[0], 1).repeat(1, q_feat.shape[1])
                # area = torch.abs(bvec_[:, :,0] * bvec_[:,:, 1] * bvec_[:,:, 2])
                # print("area",area.shape)
                # print("inp.shape", inp.shape)# [9, 12000, 75]
                pred = self.myLiif(inp.view(inp.shape[0] , inp.shape[1], -1),bvec_).view(inp.shape[0],inp.shape[1],-1)
                # print("pred", pred.shape)
                preds.append(pred)
                areas.append(area + 1e-9)

            tot_area = torch.stack(areas).sum(dim=0)
            # print("tot_area", tot_area.shape) # [9, 12000]


            # print("pre", pre.shape)
            ret = 0
            for pred, area in zip(preds, areas):
                # print("pred.shape", pred.shape)
                ret = ret + pred * ((area / tot_area)).unsqueeze(-1)

            # print("loss_all", loss_all)
            # print("label.max", lr.max())  # [9, 12000, 1]
            pre = ret
            pre = pre.reshape(lr.shape)

            lr = lr.to('cpu')
            pre = pre.to('cpu')

            pre = pre.reshape(lr.shape[0], lr.shape[1], lr.shape[2])
            """
            MRI图中的数值不仅仅表示灰度，还是带有其他纤维方向的信息，这里试一试不做归一化做损失
            2023.06.28
            """
            # print("pre.shape", pre.max())
            # print("lr.shape", lr.max())

            pre = pre * 5000.
            lr = lr * 5000.
            pre_ = (((pre - pre.min()) * 1.0) / ((pre.max() - pre.min()) * 1.0 + 1e-8)) * 1.0
            lr_ = (((lr - lr.min()) * 1.0) / ((lr.max() - lr.min()) * 1.0 + 1e-8)) * 1.0
            self.loss_l1 = self.recon_criterion(pre.to(self.device),
                                                lr.to(self.device)).item()  # 约束训练MLP网络学习隐式神经函数
            self.loss_l2 = self.criterion(pre.to(self.device),
                                          lr.to(self.device))  # 约束训练MLP网络学习隐式神经函数

            self.psnr = self.getPsnr(pre_.to(self.device), lr_.to(self.device)).item()

            self.SSIM = self.getSSIM(pre_.reshape(lr.shape[0], 1, lr.shape[1], lr.shape[2]).to(self.device),
                                     lr_.reshape(lr.shape[0], 1, lr.shape[1], lr.shape[2]).to(self.device)).item()

            total_loss = self.loss_l1
            # total_loss = self.loss_l2

            return_dict = {}
            return_dict['pre'] = pre.detach().cpu().numpy()
            return_dict['target'] = lr.detach().cpu().numpy()
            return_dict['error'] = (lr - pre).detach().cpu().numpy()
            return_dict['condition'] =condition
        self.autoEncoder.train()
        self.myLiif.train()
        return return_dict, total_loss, self.psnr,self.SSIM



    def save(self,snapshot_dir, epoch,ae_name=None,is_PSNR_best=False,is_SSIM_best=False,is_Loss_best=False):
        # Save autoencoder and optimizers
        if ae_name==None:
            if  is_PSNR_best:
                ae_name = os.path.join(snapshot_dir, 'ae_psnr_best.pt')
                liif_name = os.path.join(snapshot_dir, 'liif_psnr_best.pt')
                opt_name = os.path.join(snapshot_dir, 'opt_psnr_best.pt')
            elif is_SSIM_best:
                ae_name = os.path.join(snapshot_dir, 'ae_ssim_best.pt')
                liif_name = os.path.join(snapshot_dir, 'liif_ssim_best.pt')
                opt_name = os.path.join(snapshot_dir, 'opt_ssim_best.pt')
            elif is_Loss_best:
                ae_name = os.path.join(snapshot_dir, 'ae_loss_best.pt')
                liif_name = os.path.join(snapshot_dir, 'liif_loss_best.pt')
                opt_name = os.path.join(snapshot_dir, 'opt_loss_best.pt')
            elif epoch == -1:
                ae_name = os.path.join(snapshot_dir, 'ae_latest.pt')
                liif_name = os.path.join(snapshot_dir, 'liif_latest.pt')
                opt_name = os.path.join(snapshot_dir, 'opt_latest.pt')

            else:
                ae_name = os.path.join(snapshot_dir, 'ae_epoch%d.pt' % (epoch + 1))
                liif_name = os.path.join(snapshot_dir, 'liif_epoch%d.pt' % (epoch + 1))
                opt_name = os.path.join(snapshot_dir, 'opt_epoch%d.pt' % (epoch + 1))
        else:
            opt_name = ae_name.replace('ae', 'opt')
            liif_name = ae_name.replace('ae', 'liif')

        torch.save({'model_opt': self.ae_opt.state_dict()}, opt_name)
        torch.save({'myLiif': self.myLiif.state_dict()}, liif_name)
        torch.save({'ae': self.autoEncoder.state_dict()}, ae_name)

    def resume_mlp(self,log_dir):
        load_suffix = 'psnr_best.pt'
        print('* Resume training from {}'.format('/ae_' + load_suffix))
        print(log_dir + '/liif_' + load_suffix)
        state_dict = torch.load(log_dir + '/liif_' + load_suffix, map_location=self.device)
        # myLiif = {k: v for k, v in state_dict['ae_model'].items() if 'enc' in k}
        # print(state_dict.keys())
        # print(self.myLiif.state_dict().keys())
        self.myLiif.load_state_dict(state_dict['myLiif'])

        state_dict = torch.load(log_dir + '/opt_' + load_suffix, map_location=self.device)

        print(state_dict['model_opt']['param_groups'])
        print(self.ae_opt.state_dict()['param_groups'])
        # print("state_dict: ", str(state_dict['model_opt']))
        # print("ae_opt: ", str(self.ae_opt.state_dict()))
        self.ae_opt.load_state_dict(state_dict['model_opt'])

        state_dict = torch.load(log_dir + '/ae_' + load_suffix, map_location=self.device)
        self.autoEncoder.load_state_dict(state_dict['ae'])


    def resume(self,log_dir):
        load_suffix = 'best.pt'
        print('* Resume training from {}'.format('/ae_' + load_suffix))
        print(log_dir + '/ae_' + load_suffix)
        state_dict = torch.load(log_dir + '/ae_' + load_suffix,map_location=self.device)
        # trainer.autoEncoder.enc.load_state_dict({k: v for k, v in state_dict['ae_model'].items() if k in trainer.autoEncoder.enc.state_dict()})
        # trainer.autoEncoder.dec.load_state_dict({k: v for k, v in state_dict['ae_model'].items() if k in trainer.autoEncoder.dec.state_dict()})
        pretrained_enc = {k[4:]: v for k, v in state_dict['ae_model'].items() if 'enc' in k}
        pretrained_dec = {k[4:]: v for k, v in state_dict['ae_model'].items() if 'dec' in k}
        # print(self.autoEncoder.state_dict().keys())
        # print( '1000:',pretrained_dec.keys())
        print('1555',self.autoEncoder.enc.state_dict().keys())
        # trainer.autoEncoder.layers += torch.nn.Linear(3000, config['96000']) # 512为原始fc的数目，5是自己任务的分类数

        # trainer.autoEncoder.dec.load_state_dict(state_dict['ae_model'])
        # opt_dict = torch.load(log_dir + '/opt_' + load_suffix)
        # print(state_dict['ae_model'].items())
        # trainer.ae_opt.load_state_dict(opt_dict['ae_opt'])
        # 冻结encoder + decoder层的参数

        self.autoEncoder.enc.load_state_dict(pretrained_enc)
        # self.autoEncoder.dec.load_state_dict(pretrained_dec)
        for name, param in self.autoEncoder.named_parameters():
            if "enc" or "dec" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True