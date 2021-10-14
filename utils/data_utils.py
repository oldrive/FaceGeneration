"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


class Images_Dataset(Dataset):
    '''获取训练数据集
    得到的图片是(c, h, w)且值的大小处于[0.0, 1.0]
    '''

    def __init__(self, images_root, transforms=None):
        self.images_paths = self.make_dataset(images_root)
        self.transforms = transforms

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        images = Image.open(self.images_paths[index])
        if self.transforms:
            images = self.transforms(images)

        return images

    def make_dataset(self, dir):
        data = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                path = os.path.join(root, fname)
                data.append(path)
        return data


def cycle_images_to_create_diff_order(images, opts):
    '''
        输入：images=[img1, img2, img3]；
        输出：different_images=[img3, img1, img2]
        目的是让I_id != I_att
    '''
    batch_size = len(images)
    different_images = torch.empty_like(images, device=opts.device)
    different_images[0] = images[batch_size - 1]
    different_images[1:] = images[:batch_size - 1]
    return different_images


if __name__ == '__main__':
    from configs import data_config
    from utils.common import tensor2im, plot_image

    dataset_args = data_config.DATASETS['train_with_ffhq']
    train_dataset = Images_Dataset(dataset_args['train_images_root'],
                                   dataset_args['transform'])
    for image in train_dataset:
        print(image.shape)
        print(image)
        # image = tensor2im((image * 2) - 1)
        # plot_image(image, 'ceshi')
        break










