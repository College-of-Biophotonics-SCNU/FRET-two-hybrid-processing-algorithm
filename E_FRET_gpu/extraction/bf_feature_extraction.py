"""
对于明场图像进行特征提取操作
"""
import cv2
import numpy as np
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
from PIL import Image


dir_path = r'D:\data\20240803\MCF7-A199-5h\4'
bf_image_path = dir_path + '/BF_1.tif'
mask_image_path = dir_path + '/mask_img.png'
# 明场图像
bf_image = Image.open(bf_image_path)
bf_image_np = np.array(bf_image)

# 掩码图像
mask_image = Image.open(mask_image_path)
mask_image_np = np.array(mask_image)

# 归一化16位图像到0-255范围
bf_image_normalized = ((bf_image_np - bf_image_np.min()) / (bf_image_np.max() - bf_image_np.min()) * 255).astype(np.uint8)

# 提取每个细胞区域的属性
unique_labels = np.unique(mask_image_np)
# 去除背景标签0
unique_labels = unique_labels[unique_labels != 0]

# 存储每个细胞的纹理熵
cell_entropies = []

# 计算每个细胞的纹理熵
for label in unique_labels:
    # 获取细胞区域的掩码
    cell_mask = (mask_image_np == label)
    # 获取细胞区域的边界框
    rows, cols = np.where(cell_mask)
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)

    # 提取细胞区域
    cell_region = bf_image_normalized[min_row:max_row + 1, min_col:max_col + 1]

    # 计算GLCM
    cell_glcm = graycomatrix(cell_region, distances=[10], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, symmetric=True, normed=True)
    cell_entropy_sum = 0
    # 提取GLCM矩阵
    for i in range(0, 4):
        glcm_matrix = cell_glcm[:, :, 0, i]
        # 计算熵
        glcm_matrix = glcm_matrix.flatten()
        glcm_matrix = glcm_matrix[glcm_matrix != 0]  # 去除零值
        entropy = -np.sum(glcm_matrix * np.log2(glcm_matrix))
        cell_entropy_sum += entropy
    cell_entropies.append(cell_entropy_sum / 4)

# 打印每个细胞的纹理熵
for i, entropy in enumerate(cell_entropies):
    print(f"Cell {i+1} Entropy: {entropy}")




