import os

import numpy as np
import tifffile
import torch
from E_FRET_gpu.extraction.efficiency_feature_extraction import EFeatureExtraction
from E_FRET_gpu.segmentation.aggregates_segmentation import threshold_calculate_otsu, \
    threshold_calculate_otsu_with_single_cell_region, watershed_segmentation, region_growth_segmentation, threshold_multi_otsu_with_cell_region
from E_FRET_gpu.tool.draw_plt import draw_single


class EGFRFeatureExtraction(EFeatureExtraction):
    """
    EGFR 靶点特征提取算法
    1. 需要利用 DD 通道算法进行计算聚点定位信息
    2. 基于 DD 的共定位信息获取 Ed 图像的周围效率信息
    """
    def __init__(self, path):
        super().__init__(path)
        # 聚点分割图像
        self.image_agg = None

    def start(self):
        self.target_efficiency_extraction()
        return self.Ed_df

    def target_efficiency_extraction(self):
        """
        EGFR 靶点效率提取
        """
        print("特征提取处理 ===> ", self.image_set_path)
        # image_agg返回的是numpy格式图像
        image_agg = region_growth_segmentation(self.image_DD, self.image_mask)
        # 保存对应的图像
        tifffile.imwrite(os.path.join(self.dir_path, 'Ed_agg_img.tiff'), image_agg * self.image_Ed.numpy())
        # cv2.imwrite(os.path.join(self.dir_path, 'agg_img.jpg'), image_agg * 255)
        # 记录图像
        self.image_agg = torch.from_numpy(image_agg)
        # 创建对应的参数特征
        self.Ed_df['Ed_agg_mean_value'] = np.nan
        # 提取聚点效率值
        for label in range(1, int(self.image_mask.max()) + 1):
            Ed_label_img = self.image_Ed[self.image_mask == label]
            agg_label_img = self.image_agg[self.image_mask == label]
            Ed_agg_label_img = Ed_label_img * agg_label_img
            Ed_agg_label_tensor = Ed_agg_label_img[Ed_agg_label_img > 0]
            Ed_agg_label_tensor = Ed_agg_label_tensor.flatten()
            if len(Ed_agg_label_tensor) == 0:
                Ed_agg_label_tensor = torch.tensor([0], dtype=torch.float)
            self.Ed_df.loc[self.Ed_df['ObjectNumber'] == label, 'Ed_agg_mean_value'] = float(Ed_agg_label_tensor.mean())
            self.Ed_df.loc[self.Ed_df['ObjectNumber'] == label, 'Ed_agg_max_value'] = float(Ed_agg_label_tensor.max())
            self.Ed_df.loc[self.Ed_df['ObjectNumber'] == label, 'Ed_agg_min_value'] = float(Ed_agg_label_tensor.min())
            self.Ed_df.loc[self.Ed_df['ObjectNumber'] == label, 'Ed_agg_top_50_value'] = float(mean_of_top_half(Ed_agg_label_tensor, 1 / 2))
            self.Ed_df.loc[self.Ed_df['ObjectNumber'] == label, 'Ed_agg_top_25_value'] = float(mean_of_top_half(Ed_agg_label_tensor, 1 / 4))
        # 保存对应的结果到 原来的csv文件中
        self.save_Ed_csv()


def mean_of_top_half(tensor, proportion=1/2):
    """
    计算张量中前百分之50数据的平均值。

    :param tensor: 输入的 PyTorch 张量
    :param proportion: 比例
    :return: 前百分之50数据的平均值
    """
    # 确保输入是一个一维张量
    if tensor.dim() > 1:
        tensor = tensor.view(-1)  # 展平张量
    # 获取张量的长度
    length = tensor.size(0)
    # 计算前百分之50的数据数量
    top_k = max(1, int(length * proportion))  # 确保至少有一个元素被选中
    # 排序并选择前 top_k 个元素
    _, indices = torch.topk(tensor, top_k, largest=True)  # 选择最小的 top_k 个元素
    top_half = torch.index_select(tensor, 0, indices)
    # 计算平均值
    mean_value = top_half.mean()
    return mean_value

if __name__ == '__main__':
    image_path = r'D:\data\qrm\EGFR-CFP+GRB2-YFP\2024.06.10-A549-SH-4H\A549-AFA-4h\16'
    model = EGFRFeatureExtraction(image_path)
    model.start()