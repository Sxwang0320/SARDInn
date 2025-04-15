import os
import h5py
import numpy as np
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs

from einops import rearrange


datasets_name = [
    "train",
    "val",
     "test"
]


save_dir = rf"../../Dataset/patch2self_denoised"
data_dir = rf"../../Dataset/denoised_data"



f0 = h5py.File(rf"{save_dir}/denoise_b0_72.h5py", "w")
f1000 = h5py.File(rf"{save_dir}/denoise_b1000_72.h5py", "w")
f2000 = h5py.File(rf"{save_dir}/denoise_b2000_72.h5py", "w")
f3000 = h5py.File(rf"{save_dir}/denoise_b3000_72.h5py", "w")

for dataset_name in datasets_name:


    datas_name = os.listdir(f"{data_dir}/{dataset_name}")
    datas_name.sort()
    slice_un =72

    d0 = f0.create_dataset(dataset_name, shape=(slice_un*len(datas_name), 18 ,145,174), chunks=(1,18, 145, 174))
    d0_bvec = f0.create_dataset(dataset_name+'bvec', shape=(slice_un*len(datas_name), 18,4), chunks=(1,18, 4))

    d1000 = f1000.create_dataset(dataset_name, shape=(slice_un * len(datas_name),90, 145, 174), chunks=(1,90, 145, 174))
    d1000_bvec = f1000.create_dataset(dataset_name+'bvec', shape=(slice_un * len(datas_name), 90,4), chunks=(1, 90,4))
    #
    d2000 = f2000.create_dataset(dataset_name, shape=(slice_un * len(datas_name),90, 145, 174), chunks=(1,90, 145, 174))
    d2000_bvec = f2000.create_dataset(dataset_name + 'bvec', shape=(slice_un*len(datas_name), 90,4), chunks=(1,90,4))

    d3000 = f3000.create_dataset(dataset_name, shape=(slice_un * len(datas_name),90, 145, 174), chunks=(1,90, 145, 174))
    d3000_bvec = f3000.create_dataset(dataset_name + 'bvec', shape=(slice_un * len(datas_name),  90,4), chunks=(1, 90,4))


    for i, data_name in enumerate(datas_name):

        data_before_path = rf"{data_dir}/{dataset_name}/{data_name}/T1w/Diffusion/data.nii.gz"
        bvals_path = rf"{data_dir}/{dataset_name}/{data_name}/T1w/Diffusion/bvals"
        bvecs_path = rf"{data_dir}/{dataset_name}/{data_name}/T1w/Diffusion/bvecs"
        data_before, affine = load_nifti(data_before_path)
        bvals, bvecs = read_bvals_bvecs(bvals_path, bvecs_path)


        b0_mask = (bvals <= 100) * (bvals >= -100)
        b1000_mask = (bvals <= 1100) * (bvals >= 900)
        b2000_mask = (bvals <= 2100) * (bvals >= 1900)
        b3000_mask = (bvals <= 3100) * (bvals >= 2900)

        bvals = bvals.reshape(-1, 1)
        bval_b0 = bvals[b0_mask]
        bvecs_b0 = bvecs[b0_mask]
        bval_b1000 = bvals[b1000_mask]
        bvecs_b1000 = bvecs[b1000_mask]
        bval_b2000 = bvals[b2000_mask]
        bvecs_b2000 = bvecs[b2000_mask]
        bval_b3000 = bvals[b3000_mask]
        bvecs_b3000 = bvecs[b3000_mask]

        b0_bvec = np.hstack((bval_b0,bvecs_b0))
        b0_bvec=np.tile(b0_bvec,(slice_un,1)).reshape(slice_un,18,4)

        b1000_bvec = np.hstack((bval_b1000, bvecs_b1000))
        b1000_bvec = np.tile(b1000_bvec,(slice_un,1)).reshape(slice_un,90,4)

        b2000_bvec = np.hstack((bval_b2000, bvecs_b2000))
        b2000_bvec = np.tile(b2000_bvec, (slice_un, 1)).reshape(slice_un,90,4)

        b3000_bvec = np.hstack((bval_b3000, bvecs_b3000))
        b3000_bvec = np.tile(b3000_bvec, (slice_un, 1)).reshape(slice_un,90,4)



        data_b0 = data_before[:, :, :, b0_mask]
        data_b1000 = data_before[:, :, :, b1000_mask]
        data_b2000 = data_before[:, :, :, b2000_mask]
        data_b3000 = data_before[:, :, :, b3000_mask]



        data_b0 = data_b0[:, :,36:108, :]
        data_b1000 = data_b1000[:, :, 36:108, :]
        data_b2000 = data_b2000[:, :, 36:108, :]
        data_b3000 = data_b3000[:, :, 36:108, :]

        data_b0 = rearrange(data_b0, "h w s g -> s g h w")
        data_b1000 = rearrange(data_b1000, "h w s g -> s g h w")
        data_b2000 = rearrange(data_b2000, "h w s g -> s g h w")
        data_b3000 = rearrange(data_b3000, "h w s g -> s g h w")

        d0[i*slice_un : i*slice_un + slice_un, ...] = data_b0
        d1000[i*slice_un : i*slice_un + slice_un, ...] = data_b1000
        d2000[i*slice_un : i*slice_un + slice_un, ...] = data_b2000
        d3000[i*slice_un : i*slice_un + slice_un, ...] = data_b3000


        d0_bvec[i*slice_un : i*slice_un + slice_un, ...] = b0_bvec
        d1000_bvec[i*slice_un : i*slice_un + slice_un, ...] = b1000_bvec
        d2000_bvec[i*slice_un : i*slice_un + slice_un, ...] = b2000_bvec
        d3000_bvec[i*slice_un : i*slice_un + slice_un, ...] = b3000_bvec


f0.close()
f1000.close()
f2000.close()
f3000.close()


