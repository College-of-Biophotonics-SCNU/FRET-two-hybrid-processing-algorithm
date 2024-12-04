import warnings
import matplotlib.pyplot as plt
import tifffile
import numpy as np
import os
from cellpose import models
from skimage import io, filters
from E_FRET_gpu.constant import target_files, mask_filename

"""
cellpose CNN神经网络单细胞分割函数
"""
# 忽略特定的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*is a low contrast image")
# 忽略特定的 FutureWarning 主要是高版本的pytorch比较严谨一点
warnings.filterwarnings("ignore", category=FutureWarning, message=".*You are using `torch.load`.*")

class SegmentationModel:
    """
    使用 cellpose CNN 网络分割细胞区域
    """
    def __init__(self, root=None, img=None, diameter=200, min_box=150, max_box=400):
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

    def start(self, add_seg_channel=None):
        self.dataloader()
        for sub_folder_path in self.matching_sub_folder_paths:
            print("处理 ===> ", sub_folder_path)
            self.process(sub_folder_path, add_seg_channel)

        self.remove_subdir_path()

    def process(self, current_image_set_path, add_seg_channel=None):
        """
        开始处理计算由多通道组合计算的细胞分割流程
        :param current_image_set_path: 当前处理的最小文件夹单位
        :param add_seg_channel: 除了DD、DA、AA三通道还需要添加的分割通道
        """
        image1 = tifffile.imread(str(os.path.join(current_image_set_path, target_files[0])))
        image2 = tifffile.imread(str(os.path.join(current_image_set_path, target_files[1])))
        image3 = tifffile.imread(str(os.path.join(current_image_set_path, target_files[2])))
        combined_image = np.stack((image1, image2, image3), axis=-1)
        # 查看是否存在需要添加的分割通道文件。一般是mit线粒体或者内质网等亚细胞器染色文件
        if add_seg_channel is not None:
            for seg_channel in add_seg_channel:
                image4_path = str(os.path.join(current_image_set_path, seg_channel + '.tif'))
                if os.path.exists(image4_path):
                    image4 = tifffile.imread(image4_path)
                else:
                    continue
                # 扩展 image4 以匹配 combined_image 的维度
                image4_expanded = np.expand_dims(image4, axis=-1)
                # 沿着最后一个轴（通常是颜色通道）连接
                combined_image = np.concatenate((combined_image, image4_expanded), axis=-1)

        self.mask_image(combined_image)

        # 计算mask中是否存在贴近图像边缘的细胞，将其除去，创建一个与掩码图像相同大小的全零矩阵，用于存储处理后的掩码
        new_mask = np.zeros_like(self.current_mask, dtype=np.uint8)
        # 获取图像的高度和宽度
        height, width = self.current_mask.shape
        cell_sum = np.max(self.current_mask)
        cell_index = 1
        # 遍历像素值从高到低
        if cell_sum >= 1:
            for value in range(1, cell_sum + 1):
                # 找到当前像素值的位置
                indices = np.where(self.current_mask == value)
                # 检查细胞区域大小是否符合要求,不符合不进行录用操作
                if len(indices[0]) < self.min_size or len(indices[0]) > self.max_size:
                    continue
                # 检查每个位置是否在图像内部（不贴近边缘） 图像边框向内切 30 个像素，
                # 如果细胞在边缘条上的话，不进行记录操作
                valid_indices = [(y, x) for y, x in zip(indices[0], indices[1]) if
                                 not ((0 <= y <= 30 or height - 30 <= y <= height - 1) or
                                      (0 <= x <= 30 or width - 30 <= y <= width - 1))]
                # 如果细胞边框在四边的值x值或者y值相等的情况，也不进行录用 TODO

                # 如果有有效的位置，将当前像素值分配给它们 贴近边缘的像素点小于300就进行采用
                if len(indices[0]) - len(valid_indices) <= 300:
                    for x, y in valid_indices:
                        new_mask[x, y] = cell_index
                    cell_index += 1
        self.current_mask = new_mask
        io.imsave(str(os.path.join(current_image_set_path, mask_filename)), self.current_mask.astype(np.uint8))

    def dataloader(self):
        """
        遍历文件下的每个路径，主要是获取 AA、DA、DD 通道数据
        :return:
        """
        for root, dirs, files in os.walk(self.root):
            for sub_folder in dirs:
                sub_folder_path = str(os.path.join(root, sub_folder))
                for target_file in target_files:
                    if os.path.exists(str(os.path.join(sub_folder_path, target_file))):
                        self.matching_sub_folder_paths.append(sub_folder_path)
                        break
        print("总共找到符合要求的文件共 ", len(self.matching_sub_folder_paths))


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

    def remove_subdir_path(self):
        self.matching_sub_folder_paths.clear()

if __name__ == "__main__":
    segmentationModel = SegmentationModel()
    segmentationModel.process('../../example/egfr/control-3')
    segmentationModel.show_mask_image()
