import random
import numpy as np
import os
import csv
from time import time
import torch.utils.data
import torch
import argparse
from tqdm import tqdm
from myMain.SARDInn import dwi_SARDInn
from myUtils.utilization import get_config,prepare_sub_folder,AverageMeter,mkdirs,convert
from myData.data_get_result import getTrainData
import warnings
import h5py

maxx=[
    5800,
    4265,
    3286
]
def main():

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
        # trainer.resume('../myAE_liif/logs/test/checkpoints')
        trainer.resume_mlp('./checkpoint')
    cun = 60
    # result = h5py.File(f"../../project/Results/myAE_SRRes/debug_MLP_5/images/bvec_diff_6/b2_result_psnr_{cun}.h5py", "w")
    result = h5py.File(f"./Results/result.h5py", "w")
    data_pre = result.create_dataset('pre', shape=(7,72,6, 145, 174),
                                     chunks=(1,1,1,  145, 174))
    data_target = result.create_dataset('target', shape=(7,72,6, 145, 174),
                                     chunks=(1,1,1,  145, 174))
    data_error = result.create_dataset('error', shape=(7,72,6,  145, 174),
                                        chunks=(1, 1,1, 145, 174))
    data_vec = result.create_dataset('bvec', shape=(7,72,6, 4),
                                       chunks=(1,1,1, 4))

    test_dataset = getTrainData(folder=dataPath, mode='val', num_grad=90, pad_size=(145,174), num_slice=72,config=config)
    data_loader_test = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=1, shuffle=False,
                                                   num_workers=num_workers,
                                                   # drop_last=True,
                                                   pin_memory=False)
    # evaluation
    print('Test Evaluation ......')
    start_test = time()
    all_psnr = AverageMeter()
    test_loss_all = AverageMeter()
    all_ssim = AverageMeter()
    all_max =AverageMeter()
    bestpsnr =0
    num=0
    test_bar = tqdm(data_loader_test,desc='Test',ncols=150)
    for it, data in enumerate(test_bar):
        lr, index_slice, bvals, bvecs, index_grad, W, slice_neb, condition = data['lr'], data['index_slice'], data['bvals'], \
                                                                         data['bvecs'], data['index_grad'], data['W'], data[
                                                                             'slice_neb'], data['condition']

        with torch.no_grad():
            test_ret, val_loss_t, psnr, ssim = trainer.sample(lr, index_slice, bvals, bvecs, index_grad, W, slice_neb,
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
            num_figs = len(imgs_vis)
            p_ = index_slice[0] // 72


            for i in range(num_figs):
                if imgs_titles[i] == 'pre':
                    tmp = imgs_vis[i]

                    data_pre[p_,index_slice[0]%72,index_grad,...]  = tmp
                elif imgs_titles[i] == 'target':
                    tmp = imgs_vis[i]

                    data_target[p_,index_slice[0]%72,index_grad,...]  = tmp
                elif imgs_titles[i] == 'error':
                    tmp = imgs_vis[i]

                    data_error[p_,index_slice[0]%72,index_grad,...]  = tmp
                elif imgs_titles[i] == 'condition':
                    tmp = imgs_vis[i]

                    data_vec[p_,index_slice[0]%72,index_grad,...]  = tmp
            num=num+1

        test_bar.set_postfix(loss=val_loss_t, psnr=psnr, ssim=ssim)
    end_test = time()

    loss_print = ''
    loss_print += ' Test_Loss_dwi: %.4f' % test_loss_all.avg
    loss_print += ' Test_psnr_dwi: %.4f' % all_psnr.avg
    loss_print += ' Test_ssim_dwi: %.4f' % all_ssim.avg
    loss_print += ' Test_avg_max_dwi: %.4f' % all_max.avg
    print('[###  epoch: %d  ### Time %.3fs/epoch  (lr:%.5f)] ' % (n_epochs,end_test - start_test,config['lr']) + loss_print)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='myConfigs/testconfig.yaml', help='Path to the config file.')
    parser.add_argument('--data_root', type=str, default='', help='Path to the data, if None, get from config files')
    parser.add_argument("--resume", type=int, default=1)
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
    trainer.to(trainer.device)
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
        dataPath = '../../Dataset/dataset_h5/denoise_b2000_72.h5py'
    main()
