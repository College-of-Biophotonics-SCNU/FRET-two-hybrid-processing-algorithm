"""
降噪算法
"""
import torch
import torch.nn.functional as F


def median_filter(input_tensor, kernel_size=5):
    """
    对输入图像进行中值滤波
    :param input_tensor: 输入的图像Tensor，形状为[B, C, H, W]
    :param kernel_size: 滤波器的大小，一个整数，表示滤波器的高度和宽度
    :return: 滤波后的图像Tensor
    """
    batch_size, channels, height, width = input_tensor.size()

    # 展开滤波器窗口
    unfolded = F.unfold(input_tensor, kernel_size, padding=kernel_size // 2, stride=1)

    unfolded = unfolded.view(batch_size, channels * kernel_size ** 2, -1)

    # 计算中值
    median_vals, _ = torch.median(unfolded, dim=1)

    # 重新折叠成图像
    output_tensor = F.fold(median_vals.squeeze(1), output_size=(height, width),
                           kernel_size=1, stride=1)

    # 升维
    output_tensor = output_tensor.unsqueeze(0)
    return output_tensor
