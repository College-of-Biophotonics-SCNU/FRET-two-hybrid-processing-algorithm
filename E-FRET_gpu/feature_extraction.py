"""
基于 Ed 效率图像进行特征提取操作
1. 获取聚点位置，采用梯度下降算法获取对应的聚点区域范围
2. 统计聚点的平均效率情况
"""

import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import threshold_otsu
from tranformer import custom_to_float32_tensor
from draw_plt import draw_compare, draw_single, save_image
from noise_reduction import median_filter


def threshold_calculate_otsu(original_img, mask_img, threshold_weight=2):
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


def threshold_calculate_otsu_with_single_cell_region(original_img, mask_img, threshold_weight=1.2):
    """
    阈值分割
    1. 先使用 mask 掩码把对应的非有效区域的值进行去除 在进行统计操作
    2. 利用大津阈值法确定明显的梯度边缘阈值
    """
    region_otsu = original_img.clone()
    for cell_region in range(1, torch.max(mask_img).int() + 1):
        original_img_valid_arr = original_img[mask_img == cell_region]
        original_img_valid_arr_np = original_img_valid_arr.squeeze().squeeze().numpy()
        # 大津法确定阈值
        threshold_otsu_value = threshold_otsu(original_img_valid_arr_np) * threshold_weight

        cell_otsu_region = torch.where(original_img_valid_arr > threshold_otsu_value,
                                  torch.tensor(1, dtype=torch.float32), torch.tensor(0, dtype=torch.float32))
        region_otsu[mask_img == cell_region] = cell_otsu_region
    return region_otsu


def threshold_calculate_mean(original_img, mask_img):
    """
    使用均值减去再进行分割
    """
    original_img_valid_arr = original_img[mask_img > 0]
    # 减去均值
    threshold_otsu_value = original_img_valid_arr.mean() * 1.3
    print(threshold_otsu_value)
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


def cell_single_region_contraction(mask_cell_only_tensor, kernel_size=3):
    """
    单细胞区域内的收缩，收缩一个像素
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


if __name__ == '__main__':
    image_path = "../example/1_A/Ed.tif"
    mask_path = "../example/1_A/mask_img.png"
    aggregate = "../example/1_A/aggregates.tif"
    # 将聚点和灰度图合并
    # 读取图像
    aggregate_image = Image.open(aggregate).convert('F')
    aggregate_image_np = np.array(aggregate_image)
    image = Image.open(image_path).convert('F')
    image_np = np.array(image)

    # 将 image_np 乘以 255 并转换为整数类型
    image_np = (image_np * 255).astype(np.uint8)
    print(aggregate_image_np.shape)
    # 创建一个 RGB 版本的图像用于输出，初始化为输入的灰度图像转换后的 RGB 形式
    output_image = np.stack((image_np, image_np, image_np), axis=-1)

    # 将 aggregate 中值为 1 的点对应在输出图像上的位置设置为红色
    red_color = [255, 0, 0]
    for i in range(aggregate_image_np.shape[0]):
        for j in range(aggregate_image_np.shape[1]):
            if aggregate_image_np[i, j] == 1:
                output_image[i, j] = red_color

    # 显示输出图像
    plt.imshow(output_image)
    plt.show()

    # 保存输出图像
    # plt.imsave('output.tif', output_image)


    # # 转换为灰度图像
    # image = Image.open(image_path).convert('F')
    # mask = Image.open(mask_path).convert('L')
    # # 定义转换操作，仅将 PIL 图像转换为 tensor，不进行归一化
    # transform = transforms.Compose([
    #     lambda img: custom_to_float32_tensor(img)
    # ])
    #
    # # 加载GPU操作
    # original_img_tensor = transform(image)
    # mask_tensor = transform(mask)
    # img_tensor = original_img_tensor
    # # 增加批次参数，升维，为 1 * 1 * 2048 * 2048
    # img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
    # mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    # print("加载Ed图像情况", img_tensor.shape, img_tensor.min(), img_tensor.max())
    # print("加载mask图像的情况", mask_tensor.shape, mask_tensor.min(), mask_tensor.max())
    #
    # # 中值滤波
    # img_tensor = median_filter(img_tensor, kernel_size=5)
    # print("降噪后的Ed图像情况", img_tensor.shape, img_tensor.min(), img_tensor.max())
    #
    # # # mask 收缩
    # # resized_mask_only = cell_single_region_contraction(mask_tensor, kernel_size=5)
    # #
    # # # 计算梯度
    # # gradient_magnitude_img = gradient_calculate(img_tensor)
    # #
    # # # mask 掩码无效边缘
    # # after_mask_gradient_magnitude_img = gradient_magnitude_img * resized_mask_only
    #
    # # 均值计算聚点区域
    # # gradient_magnitude_img = threshold_calculate_mean(original_img_tensor, mask_tensor)
    # gradient_magnitude_img = threshold_calculate_otsu_with_single_cell_region(img_tensor, mask_tensor)
    # save_image(gradient_magnitude_img, "../example/1_A/aggregates.tiff")
    # draw_single(gradient_magnitude_img)
    # #
    # # # 只计算掩码区域内的梯度
    # # cell_region_gradient = gradient_magnitude_img * resized_mask_only
    # #
    # # draw(image, cell_region_gradient.squeeze().numpy())
