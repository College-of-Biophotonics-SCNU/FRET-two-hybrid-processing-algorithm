import matplotlib.pyplot as plt
import torch
from cellpose import models
from skimage import io, filters
import tifffile
import numpy as np
import os
from constant import target_files, mask_filename
from skimage.util import img_as_ubyte
"""
cellpose 单细胞分割函数
"""


class SegmentationModel:
    def __init__(self, root=None, img=None, diameter=200, min_box=100, max_box=400):
        """
        diameter 表示cellpose识别的直径大小
        min_box 表示单细胞大小最小的像素区域大小
        max_box 表示单细胞大小最大的像素区域大小
        """
        self.matching_sub_folder_paths = []
        self.model = models.Cellpose(gpu=True)
        self.root = root
        self.current_mask = None
        self.current_img = img
        self.diameter = diameter
        self.min_size = min_box * min_box
        self.max_size = max_box * max_box

    def start(self):
        self.dataloader()
        for sub_folder_path in self.matching_sub_folder_paths:
            print("处理 ===> ", sub_folder_path)
            image1 = tifffile.imread(os.path.join(sub_folder_path, target_files[0]))
            image2 = tifffile.imread(os.path.join(sub_folder_path, target_files[1]))
            image3 = tifffile.imread(os.path.join(sub_folder_path, target_files[2]))
            combined_image = np.stack((image1, image2, image3), axis=-1)
            self.mask_image(combined_image)
            # 计算mask中是否存在贴近图像边缘的细胞，将其除去
            # 创建一个与掩码图像相同大小的全零矩阵，用于存储处理后的掩码
            new_mask = np.zeros_like(self.current_mask, dtype=np.uint8)
            # 获取图像的高度和宽度
            height, width = self.current_mask.shape
            cell_sum = np.max(self.current_mask)
            cell_index = 1
            # 遍历像素值从高到低
            for value in range(1, cell_sum + 1):
                # 找到当前像素值的位置
                indices = np.where(self.current_mask == value)
                # 检查细胞区域大小是否符合要求,不符合不进行录用操作
                if len(indices[0]) < self.min_size or len(indices[0]) > self.max_size:
                    continue
                # 检查每个位置是否在图像内部（不贴近边缘） 图像边框向内切10个像素，
                # 如果细胞在边缘条上的话，不进行记录操作
                valid_indices = [(y, x) for y, x in zip(indices[0], indices[1]) if
                                 not ((0 <= y <= 10 or height - 10 <= y <= height - 1) or
                                      (0 <= x <= 10 or width - 10 <= y <= width - 1))]
                # 如果细胞边框在四边的值x值或者y值相等的情况，也不进行录用 TODO

                # 如果有有效的位置，将当前像素值分配给它们
                if valid_indices:
                    for y, x in valid_indices:
                        new_mask[y, x] = cell_index
                    cell_index += 1
            self.current_mask = np.clip(new_mask, 0, 255)
            io.imsave(os.path.join(sub_folder_path, mask_filename), self.current_mask)

    def dataloader(self):
        """
        遍历文件下的每个路径，主要是获取 AA、DA、DD 通道数据
        :return:
        """
        for root, dirs, files in os.walk(self.root):
            for sub_folder in dirs:
                sub_folder_path = os.path.join(root, sub_folder)
                for target_file in target_files:
                    if os.path.exists(os.path.join(sub_folder_path, target_file)):
                        self.matching_sub_folder_paths.append(sub_folder_path)
                        break

    def gaussian(self):
        """
        预处理操作，进行高斯模糊处理
        :return:
        """
        return filters.gaussian(self.current_img, sigma=1.5)

    def mask_image(self, img):
        self.current_img = img
        img = self.gaussian()
        self.current_mask, flows, styles, diams = self.model.eval(img,
                                                                  diameter=self.diameter,
                                                                  channels=[2, 1, 0],
                                                                  resample=True)
        return self.current_mask, flows, styles, diams

    def show_mask_image(self):
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121)
        ax1.imshow(self.current_img)
        ax1.set_title('Original Image')
        ax2 = fig.add_subplot(122)
        ax2.imshow(self.current_mask)
        ax2.set_title('Segmented Masks')
        plt.show()


if __name__ == "__main__":
    segmentationModel = SegmentationModel(root=r'D:\data\20240716\A199-A549-14')
    segmentationModel.start()
    # segmentationModel.show_mask_image()
