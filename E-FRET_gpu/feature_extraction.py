"""
基于 Ed 效率图像进行特征提取操作
1. 获取聚点位置，采用梯度下降算法获取对应的聚点区域范围
2. 统计聚点的平均效率情况
"""
import os.path

import pandas as pd
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tranformer import custom_to_float32_tensor
from draw_plt import draw_compare, draw_single, save_image
from noise_reduction import median_filter
from aggregates_segmentation import threshold_calculate_artificially_defined_with_single_cell_region, \
    threshold_calculate_otsu_with_single_cell_region


def cell_single_region_contraction(mask_cell_only_tensor, kernel_size=3):
    """
    单细胞区域内的收缩，收缩一个算子因素
    对于 mask 掩码进行收缩
    添加 mask 掩码作为屏蔽，防止计算到单细胞边缘区域，因为存在明显的梯度差
    """
    # 将获取 mask_tensor 转化为 只包含0和1的mask
    mask_cell_only_tensor = torch.where(mask_cell_only_tensor > 0,
                                        torch.tensor(1, dtype=torch.float32),
                                        torch.tensor(0, dtype=torch.float32))
    # 创建收缩算子
    conv_kernel = torch.ones(kernel_size, kernel_size, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    contraction_mask_img = torch.nn.functional.conv2d(mask_cell_only_tensor, conv_kernel,
                                                      padding=(kernel_size - 1) // 2)
    # 将值为 9 的变为 1，其他变为 0
    result_mask = torch.where(contraction_mask_img == kernel_size * kernel_size,
                              torch.tensor(1, dtype=torch.float32),
                              torch.tensor(0, dtype=torch.float32))
    return result_mask


def count_mit_efficiency(ed_image_tensor, mask_image_tensor, mit_image_tensor, ed_pandas):
    """
    提取线粒体的效率
    """
    pass


def count_aggregation_efficiency(ed_image, mask_image, aggregate_image, ed_pandas):
    """
    统计局部特征——聚点效率：
    1. 聚点的平均效率
    2. 聚点的像素数量
    3. 聚点占比
    """
    unique_cells = torch.unique(mask_image)
    unique_cells = unique_cells[unique_cells != 0]

    for cell_id in unique_cells:
        cell_aggregate = aggregate_image[mask_image == cell_id]
        cell_image = ed_image[mask_image == cell_id]
        cell_aggregated_image = cell_image[cell_aggregate == 1]

        # 添加聚点均值结果
        if len(cell_aggregated_image) != 0:
            ed_pandas.loc[ed_pandas['index'] == int(cell_id), 'aggregates_average_value'] \
                = cell_aggregated_image.mean().item()
        else:
            ed_pandas.loc[ed_pandas['index'] == int(cell_id), 'aggregates_average_value'] = 0

        # 添加聚点像素数量
        ed_pandas.loc[ed_pandas['index'] == int(cell_id), 'aggregates_pixels_value'] = len(cell_aggregated_image)

        # 添加聚点站整个单细胞区域的比例
        ed_pandas.loc[ed_pandas['index'] == int(cell_id), 'aggregates_pixels_proportion'] = (
                len(cell_aggregated_image) / len(cell_aggregate))

        # 提取空间位置分布信息

    return ed_pandas


def count_aggregation_position(ed_image, mask_image, aggregate_image, ed_pandas):
    """
    统计局部特征——聚点空间分布：
    1. 共定位特征
    """
    pass


def extraction(experiment):
    image_path = os.path.join(experiment, "Ed.tif")
    mask_path = os.path.join(experiment, "mask_img.png")
    FRET_features_pd = pd.read_csv(os.path.join(experiment, 'cell_Ed_averages.csv'))
    # 转换为灰度图像
    image = Image.open(image_path).convert('F')
    mask = Image.open(mask_path).convert('L')
    # 定义转换操作，仅将 PIL 图像转换为 tensor，不进行归一化
    transform = transforms.Compose([
        lambda img: custom_to_float32_tensor(img)
    ])

    # 加载GPU操作
    mask_tensor = transform(mask)
    original_img_tensor = transform(image)

    # 增加批次参数，升维，为 1 * 1 * 2048 * 2048
    print("加载Ed图像情况", original_img_tensor.shape, original_img_tensor.min(), original_img_tensor.max())
    print("加载mask图像的情况", mask_tensor.shape, mask_tensor.min(), mask_tensor.max())

    # 中值滤波
    img_tensor = median_filter(original_img_tensor, kernel_size=5)
    print("降噪后的Ed图像情况", img_tensor.shape, img_tensor.min(), img_tensor.max())

    # 阈值分割聚点情况
    aggregates_img = threshold_calculate_artificially_defined_with_single_cell_region(img_tensor, mask_tensor)
    save_image(aggregates_img, os.path.join(experiment, "aggregates.tif"))

    # 特征提取操作
    FRET_features_pd = count_aggregation_efficiency(original_img_tensor, mask_tensor, aggregates_img, FRET_features_pd)
    FRET_features_pd.to_csv(os.path.join(experiment, 'cell_Ed_averages.csv'), index=False)

# if __name__ == '__main__':
#     experiment = r"D:\data\20240716\A199-A549-14\0"
#     image_path = os.path.join(experiment, "Ed.tif")
#     mask_path = os.path.join(experiment, "mask_img.png")
#
#     FRET_features_pd = pd.read_csv(os.path.join(experiment, 'cell_Ed_averages.csv'))
#
#     # 转换为灰度图像
#     image = Image.open(image_path).convert('F')
#     mask = Image.open(mask_path).convert('L')
#     # 定义转换操作，仅将 PIL 图像转换为 tensor，不进行归一化
#     transform = transforms.Compose([
#         lambda img: custom_to_float32_tensor(img)
#     ])
#
#     # 加载GPU操作
#     mask_tensor = transform(mask)
#     original_img_tensor = transform(image)
#
#     # 增加批次参数，升维，为 1 * 1 * 2048 * 2048
#     print("加载Ed图像情况", original_img_tensor.shape, original_img_tensor.min(), original_img_tensor.max())
#     print("加载mask图像的情况", mask_tensor.shape, mask_tensor.min(), mask_tensor.max())
#
#     # 中值滤波
#     img_tensor = median_filter(original_img_tensor, kernel_size=5)
#     print("降噪后的Ed图像情况", img_tensor.shape, img_tensor.min(), img_tensor.max())
#
#     # 阈值分割聚点情况
#     aggregates_img = threshold_calculate_artificially_defined_with_single_cell_region(img_tensor, mask_tensor)
#     save_image(aggregates_img, experiment + "aggregates.tif")
#     # draw_single(aggregates_img)
#
#     # aggregate_path = os.path.join(experiment + "aggregates.tif")
#     # aggregate = Image.open(aggregate_path).convert('L')
#     # aggregate_tensor = transform(aggregate)
#     # 特征提取操作
#     FRET_features_pd = count_aggregation_efficiency(original_img_tensor, mask_tensor, aggregates_img, FRET_features_pd)
#     FRET_features_pd.to_csv(os.path.join(experiment, 'cell_Ed_averages.csv'), index=False)
