"""
FRET 效率特征提取流程
"""
import os

import pandas
import torch

from E_FRET_gpu.constant import target_files, efficiency_filename, mask_filename
from E_FRET_gpu.compute import load_image_to_tensor
from E_FRET_gpu.tool.FRETException import FileMissingException

def return_target_image(image_set_path, image_name):
    target_image_path = os.path.join(image_set_path, image_name)
    if os.path.exists(target_image_path):
        return load_image_to_tensor(target_image_path)
    raise FileMissingException('缺失文件：' + image_name)



class EFeatureExtraction:
    """
    FRET 效率提取， pytorch加速运算
    1. 附着于亚细胞区域的效率提取
    2. 基于靶点聚集点的效率提取
    """
    def __init__(self, path):
        # image set对应的最小单位子文件的路径
        self.image_set_path = path
        self.image_AA = None
        self.image_DD = None
        self.image_DA = None
        self.image_Ed = None
        self.image_mask = None
        self.Ed_df = None
        self.have_gpu()
        self.dataloader()
        self.dir_path = path

    def start(self):
        """
        开始流程
        """
        pass

    def dataloader(self):
        """
        数据加载
        1. 加载三通道数据
        2. 加载Ed数据
        3. 加载mask图像数据
        """
        # 验证图像文件完整
        self.image_AA = return_target_image(self.image_set_path, target_files[0])
        self.image_DD = return_target_image(self.image_set_path, target_files[1])
        self.image_DA = return_target_image(self.image_set_path, target_files[2])
        self.image_mask = return_target_image(self.image_set_path, mask_filename)
        self.image_Ed = return_target_image(self.image_set_path, efficiency_filename)
        # 加载每个位置的 site_Ed.csv
        self.Ed_df = pandas.read_csv(os.path.join(self.image_set_path, 'site_Ed.csv'))


    def target_efficiency_extraction(self):
        """
        靶点效率提取程序
        """
        pass

    def mit_efficiency_extraction(self):
        """
        线粒体效率提取
        """
        pass

    def save_Ed_csv(self):
        self.Ed_df.to_csv(os.path.join(self.image_set_path, 'site_Ed.csv'), index=False)

    @staticmethod
    def have_gpu():
        # 检查是否有可用的 GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device
