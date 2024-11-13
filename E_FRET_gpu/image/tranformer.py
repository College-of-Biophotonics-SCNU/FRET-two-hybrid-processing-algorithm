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

