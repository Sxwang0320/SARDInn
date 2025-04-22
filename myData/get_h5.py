import os
import h5py
import numpy as np
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from myUtils.torchutils import get_uniform_downsample_indices
from einops import rearrange


datasets_name = [
    # "train",
    "val",
     # "test"
]


save_dir = rf"../../../Dataset/dataset_h5"
data_dir = rf"../../../Dataset/denoised_data"



# f0 = h5py.File(rf"{save_dir}/denoise_b0_72.h5py", "w")
f1000 = h5py.File(rf"{save_dir}/denoise_b1000_72_60.h5py", "w")
# f2000 = h5py.File(rf"{save_dir}/denoise_b2000_72.h5py", "w")
# f3000 = h5py.File(rf"{save_dir}/denoise_b3000_72.h5py", "w")

for dataset_name in datasets_name:


    datas_name = os.listdir(f"{data_dir}/{dataset_name}")
    datas_name.sort()
    slice_num =72  # 需要重建的DWI数量
    re_grad_num = 60 # 需要重建的bvec数目
    able_grad = 30  # 可用的bvec数目

    re_dwis = f1000.create_dataset(dataset_name+'re_dwi', shape=(slice_num * len(datas_name),re_grad_num, 145, 174), chunks=(1,re_grad_num, 145, 174))
    re_bvecs = f1000.create_dataset(dataset_name+'re_bvec', shape=(slice_num * len(datas_name), re_grad_num,4), chunks=(1, re_grad_num,4))

    able_dwis = f1000.create_dataset(dataset_name + 'able_dwi',shape=(slice_num * len(datas_name), able_grad, 145, 174),chunks=(1, able_grad, 145, 174))
    able_bvecs = f1000.create_dataset(dataset_name + 'able_bvec', shape=(slice_num * len(datas_name), able_grad, 4), chunks=(1, able_grad, 4))

    for i, data_name in enumerate(datas_name):

        data_before_path = rf"{data_dir}/{dataset_name}/{data_name}/T1w/Diffusion/data.nii.gz"
        bvals_path = rf"{data_dir}/{dataset_name}/{data_name}/T1w/Diffusion/bvals"
        bvecs_path = rf"{data_dir}/{dataset_name}/{data_name}/T1w/Diffusion/bvecs"
        data_before, affine = load_nifti(data_before_path)
        bvals, bvecs = read_bvals_bvecs(bvals_path, bvecs_path)


        # 取单b值数据
        b1000_mask = (bvals <= 1100) * (bvals >= 900)


        bvals = bvals.reshape(-1, 1)
        bval_b1000 = bvals[b1000_mask]
        bvecs_b1000 = bvecs[b1000_mask]
        data_b1000 = data_before[:, :, :, b1000_mask]


        # 取可用的bvec，这里我们假设在HCP数据中只有able_grad个bvec可用
        cond_list =get_uniform_downsample_indices(bvecs_b1000, able_grad,0)
        all_index = [i for i in range(0, 90)]  # 总的bvec数目
        pre_list = list(set(all_index).difference(cond_list))  # 需要重建的bvec索引



        able_bvec = bvecs_b1000[cond_list]  # 可用bvec
        able_bval = bval_b1000[cond_list]
        able_dwi = data_b1000[:, :, 36:108, cond_list] # 可用bvec的DWI
        able_bvec = np.hstack((able_bval, able_bvec))
        able_bvec = np.tile(able_bvec,(slice_num,1)).reshape(slice_num,able_grad,4)

        able_dwi = rearrange(able_dwi, "h w s g -> s g h w")


        able_dwis[i*slice_num : i*slice_num + slice_num, ...] = able_dwi
        able_bvecs[i*slice_num : i*slice_num + slice_num, ...] = able_bvec

        re_bvec = bvecs_b1000[pre_list]  # 可用bvec
        re_bval = bval_b1000[pre_list]
        re_dwi = data_b1000[:, :, 36:108, pre_list] # 可用bvec的DWI

        re_bvec = np.hstack((re_bval, re_bvec))
        re_bvec = np.tile(re_bvec,(slice_num,1)).reshape(slice_num,re_grad_num,4)

        re_dwi = rearrange(re_dwi, "h w s g -> s g h w")

        re_dwis[i * slice_num: i * slice_num + slice_num, ...] = re_dwi
        re_bvecs[i * slice_num: i * slice_num + slice_num, ...] = re_bvec


# f0.close()
f1000.close()
# f2000.close()
# f3000.close()


