import os

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.ndimage import generate_binary_structure
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from E_FRET_gpu.image.tranformer import tensor_to_8bit_np
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


def threshold_multi_otsu_with_cell_region(original_img_tensor, mask_img_tensor,
                                            need_connected_regions=True,
                                            aggregates_size=10):
    """
    使用三类Otsu算法进行阈值分割。

    :param original_img_tensor: 输入的原始图像，输入需要是二维图像
    :param mask_img_tensor: mask掩码，输入需要是二维图像
    :param need_connected_regions: 判断是否需要连通区域进行分析
    :param aggregates_size: 聚点阈值，将小于该区域像素点的阈值抛出
    :return: 分割后的二值图像
    """
    # 初始化输出图像

    original_img = tensor_to_8bit_np(original_img_tensor)
    mask_img = np.array(mask_img_tensor, dtype=np.uint8)
    region_otsu = np.zeros_like(mask_img, dtype=np.uint8)
    for cell_region in range(1, int(mask_img.max()) + 1):
        # 提取单细胞区域的有效像素值
        valid_mask = mask_img == cell_region
        original_img_valid_arr = original_img[valid_mask]
        if len(original_img_valid_arr) == 0:
            continue
        # 使用 multi-Otsu 确定两个阈值
        thresholds = threshold_multiotsu(original_img_valid_arr)
        print(f"该图像单细胞区域的灰度阈值： {thresholds}")
        # 根据两个阈值将区域分为三类
        region_otsu[valid_mask] = np.searchsorted(thresholds, original_img_valid_arr, side='right')
    # 如果需要连通区域分析，则执行此步骤
    if need_connected_regions:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(region_otsu, connectivity=8)
        # 创建一个新的numpy数组来存储筛选后的结果
        filtered_result = np.zeros_like(region_otsu)
        # 跳过背景标签0
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > aggregates_size:
                filtered_result[labels == i] = region_otsu[labels == i]
        return filtered_result
    else:
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

def region_growing(image, seeds, threshold=30, max_points=100):
    """
    执行区域生长算法，并限制最大生长点数。

    :param image: 输入图像 (numpy array)
    :param seeds: 种子点列表 [(row, col)]
    :param threshold: 生长阈值，表示允许的最大灰度差异
    :param max_points: 最大生长点数，默认为100
    :return: 区域生长后的二值图像
    """
    segmented = np.zeros_like(image, dtype=np.uint8)
    processed = np.zeros_like(image, dtype=bool)
    s = generate_binary_structure(2, 2)  # 8-连通结构

    seed_values = {seed: image[seed] for seed in seeds}  # 记录每个种子点的灰度值

    for seed in seeds:
        queue = [seed]
        processed[seed[0], seed[1]] = True
        segmented[seed[0], seed[1]] = 1
        point_count = 1
        while queue:
            current = queue.pop(0)
            current_seed_value = seed_values[seed]  # 使用种子点的灰度值作为参考

            for offset in zip(*s.nonzero()):
                x_new, y_new = current[0] + offset[0] - 1, current[1] + offset[1] - 1
                if (0 <= x_new < image.shape[0]) and (0 <= y_new < image.shape[1]):
                    neighbor_value = image[x_new, y_new]
                    if abs(int(neighbor_value) - int(current_seed_value)) < threshold:
                        if not processed[x_new, y_new]:
                            segmented[x_new, y_new] = 1
                            queue.append((x_new, y_new))
                            processed[x_new, y_new] = True
                        point_count += 1  # 更新计数器
                if point_count >= max_points:  # 达到最大点数后停止生长
                    break
    return segmented


def filter_connected_components(segmented_image, min_size=20):
    """
    筛选连通组件，移除面积小于min_size或大于max_size的组件。

    :param segmented_image: 区域生长后的二值图像
    :param min_size: 最小连通区域面积，默认20
    :return: 过滤后的二值图像
    """
    # 使用 OpenCV 找到所有连通组件及其统计信息
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented_image, connectivity=8)

    filtered_image = np.zeros_like(segmented_image, dtype=np.uint8)
    # 创建一个掩码来保存符合条件的连通区域
    mask = np.zeros_like(segmented_image, dtype=bool)

    # 如果细胞存在
    for i in range(1, num_labels):  # 跳过背景（标签0）
        if min_size <= stats[i, cv2.CC_STAT_AREA]:
            mask[labels == i] = True
    # 应用掩码生成最终结果
    filtered_image[mask] = 1

    return filtered_image

def region_growth_segmentation(original_img_tensor, mask_img_tensor):
    """
    在每个已标记的单细胞区域内进行聚点分割，并使用区域生长算法扩展聚点区域。

    算法存在问题后续进行优化操作

    :param original_img_tensor: 灰度图像 (numpy array)
    :param mask_img_tensor: 标记了单细胞区域的掩码图像 (numpy array)，其中不同细胞用不同的正整数标记
    :return: 分割后的二值图像，其中每个聚点及其扩展区域为1，其余位置为0
    """
    # 确保输入是numpy数组并转换为uint8类型
    image = np.array(original_img_tensor, dtype=np.float32)
    labeled_mask = np.array(mask_img_tensor, dtype=np.uint8)

    # 初始化输出二值图像
    binary_image = np.zeros_like(labeled_mask, dtype=np.uint8)

    # 获取所有细胞的属性
    regions = regionprops(labeled_mask)

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        cell_label = region.label
        cell_mask = (labeled_mask == cell_label)[minr:maxr, minc:maxc]
        cell_mask[cell_mask > 0] = 1
        cell_image = image[minr:maxr, minc:maxc] * cell_mask

        # 对于cell_image 进行归一化操作，方便计算
        cell_image_min = cell_image[cell_image > 0].min()
        cell_image_max = cell_image.max()

        # 单个区域在进行归一化操作 防止存在单细胞转换效率低的情况
        cell_image = 255 * (cell_image - cell_image_min) / (cell_image_max - cell_image_min)
        cell_image[cell_image < 0] = 0
        # 转换回无符号8位整数类型
        cell_image = cell_image.astype(np.uint8)

        # 预处理：高斯模糊减少噪声，增强对比度（可选）
        blurred = cv2.GaussianBlur(cell_image, (5, 5), 0)

        # 检测当前细胞内的局部最大值
        coordinates = peak_local_max(blurred, min_distance=20, threshold_abs=100)  # 调整参数以适应您的需求

        # 将坐标转换回原始图像坐标系
        coordinates[:, 0] += minr
        coordinates[:, 1] += minc

        # 创建标记图像
        markers = []
        for row, col in coordinates:
            if labeled_mask[row, col] == cell_label:  # 确保标记在该细胞区域内
                markers.append((row - minr, col - minc))

        # 应用区域生长算法
        grown_regions = region_growing(cell_image * cell_mask, markers, threshold=30)  # 调整阈值以适应您的需求
        # 连通区域分析
        filtered_grown_regions = filter_connected_components(grown_regions)
        # 更新全局二值图像
        binary_image[minr:maxr, minc:maxc][filtered_grown_regions != 0] = 1

    return binary_image

def watershed_segmentation(original_img_tensor, mask_img_tensor):
    """
    分割效果太差：分水岭算法
    """
    # 确保输入是numpy数组并转换为uint8类型
    image = tensor_to_8bit_np(original_img_tensor)

    labeled_mask = np.array(mask_img_tensor, dtype=np.uint8)

    # 初始化二值图像
    binary_image = np.zeros_like(image, dtype=np.uint8)

    # 获取所有细胞的属性
    regions = regionprops(labeled_mask)

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        cell_label = region.label
        cell_mask = (labeled_mask == cell_label)[minr:maxr, minc:maxc]
        cell_image = image[minr:maxr, minc:maxc]

        # 预处理：高斯模糊减少噪声，增强对比度（可选）
        blurred = cv2.GaussianBlur(cell_image, (5, 5), 0)

        # 检测当前细胞内的局部最大值
        coordinates = peak_local_max(blurred, min_distance=20, threshold_abs=100)  # 调整参数以适应您的需求
        # 将坐标转换回原始图像坐标系
        coordinates[:, 0] += minr
        coordinates[:, 1] += minc
        # 创建标记图像
        markers = np.zeros_like(cell_mask, dtype=np.int32)
        for i, (row, col) in enumerate(coordinates):
            if labeled_mask[row, col] == cell_label:  # 确保标记在该细胞区域内
                markers[row - minr, col - minc] = i + 1  # 给每个局部最大值分配一个唯一的标记编号
        # 距离变换（可选，取决于是否需要更精确的前景估计）
        distance = ndi.distance_transform_edt(cell_mask)
        # 应用分水岭算法
        cell_labels = watershed(-distance, markers, mask=cell_mask)
        # 更新全局二值图像，仅保留聚点区域
        for i in range(1, len(coordinates) + 1):
            spot_region = cell_labels == i
            binary_image[minr:maxr, minc:maxc][spot_region] = 1
    return binary_image


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
