import cv2
import numpy as np
import torch
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


def threshold_calculate_otsu(original_img, mask_img, threshold_weight=1.5):
    """
    阈值分割
    1. 先使用 mask 掩码把对应的非有效区域的值进行去除 在进行统计操作
    2. 利用大津阈值法确定明显的梯度边缘阈值
    """
    original_img_valid_arr = original_img[mask_img > 0]
    original_img_valid_arr = original_img_valid_arr.squeeze().squeeze().numpy()

    # 大津法确定阈值
    threshold_otsu_value = threshold_otsu(original_img_valid_arr) * threshold_weight

    region_otsu = torch.where(original_img > threshold_otsu_value,
                              torch.tensor(1, dtype=torch.float32), torch.tensor(0, dtype=torch.float32))

    return region_otsu


def threshold_calculate_otsu_with_single_cell_region(original_img, mask_img, threshold_weight=1.2,
                                                     need_connected_regions=True, aggregates_size=10):
    """
    阈值分割

    :parma original_img: 输入的原始图像，输入需要是二维图像
    :param mask_img: mask掩码，输入需要是二维图像
    :param threshold_weight: 阈值比例，给ostu算法得出的阈值添加一个权重
    :param need_connected_regions: 判断是否需要连通区域进行分析
    :param aggregates_size: 聚点阈值，将小于该区域像素点的阈值抛出
    1. 先使用 mask 掩码把对应的单细胞区域圈出来
    2. 利用大津阈值法确定明显的梯度边缘阈值
    """
    original_img = original_img
    region_otsu = np.zeros_like(mask_img, dtype=np.uint8)
    for cell_region in range(1, torch.max(mask_img).int() + 1):
        original_img_valid_arr = original_img[mask_img == cell_region]
        original_img_valid_arr_np = original_img_valid_arr.numpy()
        # 大津法确定阈值
        threshold_otsu_value = threshold_otsu(original_img_valid_arr_np) * threshold_weight
        print("该图像单细胞区域的灰度阈值： ", threshold_otsu_value)
        cell_otsu_region = torch.where(original_img_valid_arr > threshold_otsu_value,
                                       torch.tensor(1, dtype=torch.int8), torch.tensor(0, dtype=torch.int8))
        region_otsu[mask_img == cell_region] = cell_otsu_region

    # 使用 OpenCV 的 connectedComponentsWithStats 函数
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(region_otsu, connectivity=8)

    # 创建一个新的numpy数组来存储筛选后的结果
    filtered_result = np.zeros_like(region_otsu)
    # 跳过背景标签0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > aggregates_size:
            filtered_result[labels == i] = 1
    return filtered_result


def threshold_calculate_artificially_defined_with_single_cell_region(original_img,
                                                                     mask_img,
                                                                     threshold_value=0.3,
                                                                     aggregates_size=20):
    """
    人为定义阈值进行有效效率分割操作
    """
    region_otsu = torch.where(original_img > threshold_value,
                              torch.tensor(1, dtype=torch.float32), torch.tensor(0, dtype=torch.float32))

    # 使用 skimage 的 label 函数进行连通区域标记
    labeled_array, num_features = label(region_otsu, return_num=True)

    # 遍历每个连通区域，筛选出面积大于 aggregates_size 的区域
    filtered_regions = []
    for region in regionprops(labeled_array):
        if region.area > aggregates_size:
            filtered_regions.append(region)

    # 创建一个新的tensor来存储筛选后的结果
    filtered_result = torch.zeros_like(region_otsu)
    for region in filtered_regions:
        coords = region.coords
        for coord in coords:
            filtered_result[coord[0], coord[1]] = 1
    return filtered_result


def threshold_calculate_mean(original_img, mask_img):
    """
    使用均值减去再进行分割
    """
    original_img_valid_arr = original_img[mask_img > 0]
    # 减去均值
    threshold_otsu_value = original_img_valid_arr.mean() * 1.3
    region_otsu = torch.where(original_img > threshold_otsu_value,
                              torch.tensor(1, dtype=torch.float32), torch.tensor(0, dtype=torch.float32))
    return region_otsu


def gradient_calculate(image_tensor):
    """
    梯度计算
    """
    # Sobel 梯度算子
    sobel_filter_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_filter_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    grad_x = torch.nn.functional.conv2d(image_tensor, sobel_filter_x, padding=1)
    grad_y = torch.nn.functional.conv2d(image_tensor, sobel_filter_y, padding=1)

    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return gradient_magnitude
