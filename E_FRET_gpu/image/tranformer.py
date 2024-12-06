import numpy as np
import torch
"""
tensor 加载转换工具类
1. 转为 int 8 位函数
2. 转为 float 32 位函数
"""

def custom_to_unit8_tensor(image):
    img_tensor = torch.tensor(list(image.getdata()), dtype=torch.uint8).reshape(image.height, image.width, -1)
    if img_tensor.shape[2] == 1:
        img_tensor = img_tensor.squeeze(2)
    return img_tensor


def custom_to_float32_tensor(image):
    img_tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(image.height, image.width, -1)
    if img_tensor.shape[2] == 1:
        img_tensor = img_tensor.squeeze(2)
    return img_tensor


def tensor_to_8bit_np(tensor):
    # 确保张量在CPU上
    tensor = tensor.cpu()
    min_val = tensor.min()
    max_val = tensor.max()
    # 归一化并缩放到0-255
    tensor = 255 * (tensor - min_val) / (max_val - min_val)
    # 转换为NumPy数组并裁剪值
    np_img = tensor.numpy().clip(0, 255)
    # 转换为8-bit无符号整数
    return np_img.astype(np.uint8)