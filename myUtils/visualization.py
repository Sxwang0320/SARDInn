import numpy as np
import matplotlib.pyplot as plt
import os



def tensorboard_vis(epoch,step, board_name,img_list,titles,image_directory):
    """
    :param summarywriter: tensorboard summary writer handle
    :param step:
    :param board_name: display name
    :param img_list: a list of images to show
    :param num_row: specify the number of row
    :param cmaps: specify color maps for each image ('gray' for MRI, i.e.)
    :param titles: specify each image title
    :param resize: whether resize the image to show
    :return:
    """

    pre_path = image_directory
    pre_path = os.path.join(pre_path, board_name)
    if os.path.exists(pre_path):
        pass
    else:
        os.mkdir(pre_path)

    pre_path = os.path.join(pre_path, str(epoch))

    if os.path.exists(pre_path):
        pass
    else:
        os.mkdir(pre_path)

    pre_path = os.path.join(pre_path, str(step))

    if os.path.exists(pre_path):
        pass
    else:
        os.mkdir(pre_path)


    num_figs = len(img_list)
    for i in range(num_figs):
        if titles[i] == 'pre':
            tmp = img_list[i]
            # tmp = (((tmp - tmp.min()) * 255.0) / ((tmp.max() - tmp.min()) * 255.0 + 1e-8)) * 255.0
            # tmp = tmp.astype(np.uint8)
            # print('pre',tmp.max())
            plt.figure()
            plt.imshow(tmp, cmap='YlGnBu')
            plt.colorbar(label='Error Magnitude (log scale)')
            plt.title('pre Image')
            plt.savefig(os.path.join(pre_path, 'pre.png'))
            # imageio.imwrite(
            #     os.path.join(pre_path, 'pre.png'), tmp)  # 保存图像
        elif titles[i] == 'target':
            tmp = img_list[i]
            # tmp = (((tmp - tmp.min()) * 255.0) / ((tmp.max() - tmp.min()) * 255.0 + 1e-8)) * 255.0
            # tmp = tmp.astype(np.uint8)
            # print('tar', tmp.max())
            plt.figure()
            plt.imshow(tmp, cmap='YlGnBu')
            plt.colorbar(label='Error Magnitude (log scale)')
            plt.title('tar Image')
            plt.savefig(os.path.join(pre_path, 'target.png'))
            # imageio.imwrite(
            #     os.path.join(pre_path, 'target.png'), tmp)  # 保存图像
        elif titles[i] == 'error':
            tmp = img_list[i]
            # tmp = (((tmp - tmp.min()) * 255.0) / ((tmp.max() - tmp.min()) * 255.0 + 1e-8)) * 255.0
            # tmp = tmp.astype(np.uint8)
            error_magnitude = np.log10(tmp.max())
            plt.figure()
            plt.imshow(tmp, cmap='YlGnBu')
            plt.colorbar(label='Error Magnitude (log scale)')
            plt.title('Error Image')
            plt.savefig(os.path.join(pre_path, 'error.png'))
            # imageio.imwrite(
            #     os.path.join(pre_path, 'error.png'), tmp)  # 保存图像

    return








