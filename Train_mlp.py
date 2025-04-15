import random
import numpy as np
import os
import torch.nn as nn
import csv
import json
from time import time
import torch.utils.data
import torch
import argparse
from torch import optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from myMain.SARDInn import dwi_SARDInn
from myUtils.visualization import tensorboard_vis
from myUtils.utilization import get_config,prepare_sub_folder,AverageMeter,mkdirs,convert
from myUtils.torchutils import adjust_learning_rate
from myData.data_loader import getTrainData,getTestANDValData


def main():
    train_dataset = getTrainData(folder=dataPath, mode='train', num_grad=num_grad,  pad_size=pad_size,num_slice=72, config=config)
    val_dataset = getTestANDValData(folder=dataPath, mode='val', num_grad=num_grad,  pad_size=pad_size,num_slice=72, config=config)


    data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size, shuffle=True,
                                                    num_workers=num_workers,
                                                    drop_last=True,
                                                    pin_memory=False)
    data_loader_val = torch.utils.data.DataLoader(dataset=val_dataset,
                                                  batch_size=8, shuffle=False,
                                                  num_workers=num_workers,
                                                  drop_last=True,
                                                  pin_memory=False)


    # 这里已经加载数据进入显存了，可能是模型数据
    print('---------------------------------------------------------------------')

    #################################
    # Setup logger and output folders
    #################################

    log_dir = config['log_dir']
    # Setup directories
    output_directory = os.path.join(config['log_dir'], config['experiment_name'])
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    if not os.path.exists(log_dir):
        print('* Creating log directory: ' + log_dir)
        mkdirs(log_dir)
    print('* Logs will be saved under: ' + log_dir)

    # json.dumps() 方法 将一个Python数据结构转换为JSON字符串
    options_str = json.dumps(config, indent=4, sort_keys=False)
    # 以JSON的格式来存储模型超参数
    with open(os.path.join(output_directory, 'options.json'), 'w') as f:
        f.write(options_str)
    with open(os.path.join(output_directory, 'train_loss.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'psnr', 'ssim'])
    with open(os.path.join(output_directory, 'val_loss.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'psnr', 'ssim'])
    with open(os.path.join(output_directory, 'test_loss.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(['loss', 'psnr', 'ssim'])

    # Start training
    # 加载预训练
    if config['pretrained'] != '':
        trainer.resume(config['pretrained'])

    load_epoch = -1

    # resume大于0就是恢复训练
    if opts.resume > 0:
        # trainer.resume('../AE_pre/logs/resblocks_pre_b1_dim_72/checkpoints')
        # trainer.resume('../test_myAE_gft_fuben/get_pre/resblocks_pre_b1/checkpoints')
        #
        #trainer.resume_mlp('../myAE_gft/logs/train_wsx_29_3/checkpoints')
        trainer.resume('../myAE_gft/result/pre_train_4/checkpoints')



    ep0 = load_epoch
    print('* Training from epoch %d to %d' % (ep0 + 1, n_epochs))
    print('---------------------------------------------------------------------')
    ep0 += 1
    start = time()
    train_loss = []
    train_psnr_all = []
    train_ssim_all = []
    val_loss = []
    val_psnr_all = []
    val_ssim_all = []
    best_loss = 10000000
    best_psnr=0
    best_ssim=0
    for epoch in range(ep0,n_epochs):
        trainer.train()
        train_loss_all =AverageMeter()
        all_psnr = AverageMeter()
        all_ssim = AverageMeter()
        adjust_learning_rate(trainer.ae_opt,epoch,config)
        # ncols 设置进度条宽度
        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size),ncols=150) as t:
        # 设置进度条信息通过set_description()设置进度条左边显示信息和set_postfix()设置进度条右边显示信息
            t.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            for it ,data in enumerate(data_loader_train):

                lr, index_slice,bvals,bvecs,sizeof,W,slice_neb = data['lr'], data['index_slice'], data['bvals'], data['bvecs'], data['sizeof'],data['W'],data['slice_neb']
                # print("sizeof",sizeof.shape)
                train_ret,loss,train_psnr,train_ssim = trainer.forward(lr, index_slice,bvals,bvecs,sizeof,W,slice_neb)
                # flops, params = profile(trainer, inputs=(lr, index_slice,bvals,bvecs,sizeof,W,slice_neb))
                # print("flops: ", flops, "  params: ", params)
                train_loss_all.update(loss)
                all_psnr.update(train_psnr)
                all_ssim.update(train_ssim)
                # 保存训练图像
                if it == config['log_freq_train']:
                    imgs_vis = [train_ret[k] for k in train_ret.keys()]
                    imgs_titles = list(train_ret.keys())
                    cmaps = ['jet' if 'seg' in i else 'gist_gray' for i in imgs_titles]
                    # print('iterations', iterations)
                    tensorboard_vis(epoch=epoch + 1, step=it, board_name='train',img_list=imgs_vis, titles=imgs_titles, image_directory=image_directory)
                t.set_postfix(loss=loss,psnr=train_psnr,ssim=train_ssim)
                t.update(len(lr))

            t.set_postfix(avg_loss=train_loss_all.avg,avg_psnr=all_psnr.avg,avg_ssim=all_ssim.avg)
        train_loss.append(train_loss_all.avg)
        train_psnr_all.append(all_psnr.avg)
        train_ssim_all.append(all_ssim.avg)

        # trainer.scheduler.step(train_loss_all.avg)# 动态修改学习率

        with open(os.path.join(output_directory, 'train_loss.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch,train_loss_all.avg,all_psnr.avg,all_ssim.avg])

        # save checkpoint
        if (epoch + 1) <10:
            trainer.save(checkpoint_directory, epoch)


        if (epoch + 1) % config['eval_epochs']==0:
            # evaluation
            print('Validation Evaluation ......')
            trainer.eval()
            with torch.no_grad():

                avg_psnr_v = AverageMeter()
                avg_ssim_v = AverageMeter()
                avg_loss_v = AverageMeter()
                start_t = time()
                val_bar = tqdm(data_loader_val,desc='VAL',ncols=150)
                for it,data in enumerate(val_bar):
                    lr, index_slice,bvals,bvecs,sizeof,W,slice_neb,condition = data['lr'], data['index_slice'], data['bvals'], data['bvecs'], data['sizeof'],data['W'],data['slice_neb'],data['condition']
                    test_ret, val_loss_t, psnr, ssim = trainer.sample(lr, index_slice,bvals,bvecs,sizeof,W,slice_neb,condition)

                    avg_psnr_v.update(psnr)
                    avg_ssim_v.update(ssim)
                    avg_loss_v.update(val_loss_t)

                    # 验证保存结果
                    if (it+1) == config['save_val_it']:
                        imgs_vis = [test_ret[k] for k in test_ret.keys()]
                        imgs_titles = list(test_ret.keys())
                        cmaps = ['jet' if 'seg' in i else 'gist_gray' for i in imgs_titles]
                        # tensorboard_vis(epoch=epoch+1,step=it, board_name='val',img_list=imgs_vis,titles=imgs_titles,image_directory=image_directory)
                    val_bar.set_postfix(loss=val_loss_t, psnr=psnr, ssim=ssim)
                end_t = time()
                update_t = end_t - start_t
                loss_print = ''
                loss_print += ' Psnr_dwi: %.4f' % avg_psnr_v.avg
                loss_print += ' ssim: %.4f' % avg_ssim_v.avg
                loss_print += ' Loss: %.4f' % avg_loss_v.avg
                print('[###  VaL  ###:Time %.3fs/it  epcoh: %d (lr:%.5f)] ' % (update_t, epoch, trainer.ae_opt.param_groups[1]['lr']) + loss_print)

                with open(os.path.join(output_directory, 'val_loss.csv'), 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, avg_loss_v.avg, avg_psnr_v.avg,avg_ssim_v.avg])


                if avg_psnr_v.avg > best_psnr:

                    best_psnr = avg_psnr_v.avg
                    trainer.save(checkpoint_directory, epoch, is_PSNR_best=True)
                    with open(checkpoint_directory + '/best_psnr_log.txt', 'w') as f:
                        f.writelines('%d' % (epoch))
                if avg_ssim_v.avg > best_ssim:
                    best_ssim = avg_ssim_v.avg
                    trainer.save(checkpoint_directory, epoch, is_SSIM_best=True)
                    with open(checkpoint_directory + '/best_ssim_log.txt', 'w') as f:
                        f.writelines('%d' % (epoch))
                if avg_loss_v.avg < best_loss:
                    best_loss = avg_loss_v.avg
                    trainer.save(checkpoint_directory, epoch, is_Loss_best=True)
                    with open(checkpoint_directory + '/best_loss_log.txt', 'w') as f:
                        f.writelines('%d' % (epoch))
                val_loss.append(avg_loss_v.avg)
                val_psnr_all.append(avg_psnr_v.avg)
                val_ssim_all.append(avg_ssim_v.avg)
                # 保存最好的模型参数



    end = time()
    print('Training finished in {}, {} epochs'.format(convert(end - start), n_epochs))
    train_process = pd.DataFrame(
    data={"epoch":range(n_epochs),
          "train_loss":train_loss,
          "train_psnr":train_psnr_all,
          "train_ssim":train_ssim_all,
         }
    )
    val_process = pd.DataFrame(
        data={
            "epoch": range(n_epochs),
            "val_loss":val_loss,
            "val_psnr":val_psnr_all,
            "val_ssim":val_ssim_all,
        }
    )
    plt.figure()
    plt.subplot(1,1,1)
    plt.plot(train_process['epoch'], train_process.train_loss, "ro-", label="Train loss")
    plt.xlabel("epoch")
    plt.ylabel("Train loss")
    plt.legend()
    plt.savefig(image_directory + '/train_loss.png')
    plt.show()
    plt.subplot(1, 1, 1)
    plt.plot(val_process['epoch'], val_process.val_loss, "bs-", label="Val loss")
    plt.xlabel("epoch")
    plt.ylabel("Val loss")
    plt.legend()
    plt.savefig(image_directory + '/val_loss.png')
    plt.show()
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot(train_process['epoch'], train_process.train_psnr, "ro-", label="train_psnr")
    plt.plot(val_process['epoch'], val_process.val_psnr, "bs-", label="val_psnr")
    plt.xlabel("epoch")
    plt.ylabel("psnr")
    plt.legend()
    plt.savefig(image_directory + '/psnr.png')
    plt.show()
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot(train_process['epoch'], train_process.train_ssim, "ro-", label="train_ssim")
    plt.plot(val_process['epoch'], val_process.val_ssim, "bs-", label="val_ssim")
    plt.xlabel("epoch")
    plt.ylabel("ssim")
    plt.legend()
    plt.savefig(image_directory + '/ssim.png')
    plt.show()

    test_dataset = getTestANDValData(folder=dataPath, mode='test', num_grad=num_grad, pad_size=pad_size, num_slice=16,
                                     config=config)
    data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=1, shuffle=False,
                                                   num_workers=num_workers,
                                                   drop_last=True,
                                                   pin_memory=False)
    # evaluation
    print('Test Evaluation ......')
    start_test = time()
    all_psnr = AverageMeter()
    test_loss_all = AverageMeter()
    all_ssim = AverageMeter()
    test_bar = tqdm(data_loader_test,desc='Test',ncols=150)
    for it, data in enumerate(test_bar):
        lr, index_slice, bvals, bvecs, sizeof, W, slice_neb, condition = data['lr'], data['index_slice'], data['bvals'], \
                                                                         data['bvecs'], data['sizeof'], data['W'], data[
                                                                             'slice_neb'], data['condition']

        with torch.no_grad():
            test_ret, val_loss_t, psnr, ssim = trainer.sample(lr, index_slice, bvals, bvecs, sizeof, W, slice_neb,
                                                              condition)
            all_psnr.update(psnr)
            all_ssim.update(ssim)
            test_loss_all.update(val_loss_t)
            imgs_vis = [test_ret[k] for k in test_ret.keys()]
            imgs_titles = list(test_ret.keys())
            cmaps = ['jet' if 'seg' in i else 'gist_gray' for i in imgs_titles]
            with open(os.path.join(output_directory, 'test_loss.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow([str(val_loss_t),str(psnr),str(ssim)])

            tensorboard_vis(epoch=n_epochs,step=it, board_name='test',img_list=imgs_vis,titles=imgs_titles,image_directory=image_directory)
        test_bar.set_postfix(loss=val_loss_t, psnr=psnr, ssim=ssim)
    end_test = time()
    loss_print = ''
    loss_print += ' Test_Loss_dwi: %.4f' % test_loss_all.avg
    loss_print += ' Test_psnr_dwi: %.4f' % all_psnr.avg
    loss_print += ' Test_ssim_dwi: %.4f' % all_ssim.avg
    print('[###  epoch: %d  ### Time %.3fs/epoch  (lr:%.5f)] ' % (n_epochs,end_test - start_test,config['lr']) + loss_print)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='myConfigs/configs.yaml', help='Path to the config file.')
    parser.add_argument('--data_root', type=str, default='', help='Path to the data, if None, get from config files')
    parser.add_argument("--resume", type=int, default=0)
    opts = parser.parse_args()

    ##########################
    # Load experiment setting
    ##########################

    config = get_config(opts.config)
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    num_grad = config['num_grad']
    pad_size = config['pad_size']
    random.seed(config['seed'])
    random.seed(config['seed'])
    os.environ['PYTHONHASHSEED'] = str(config['seed'])  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])  # if you are using multi-GPU.

    torch.cuda.manual_seed_all(config['seed'])
    n_epochs = config['n_epoch']
    ## Load data ##
    local = config['local']

    #############################
    # Setup model and data loader
    #############################

    print('* Set up models ...')
    print('---------------------------------------------------------------------')

    trainer = dwi_SARDInn(config)

    # trainer.to(trainer.device)
     # 在执行该语句之前最好加上model.cuda(),保证你的模型存在GPU上即可

    print('---------------------------------------------------------------------')
    print('* Dataset ')
    print('---------------------------------------------------------------------')
    print('- ' + config['dataset'])

    if opts.data_root == '':
        data_root = config['data_root']
    else:
        data_root = opts.data_root
    print('* Data root defined in ' + data_root)
    if local:
        dataPath = '/home/hkebiri/Desktop/DHCP_Preterm37_Data'
    else:
        dataPath = '/home/imiapd/wangshuangxing22/Dataset/dataset_h5/denoise_b1000_72.h5py'
    main()
