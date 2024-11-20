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
    input_tensor = input_tensor.unsqueeze(0)
    channels, height, width = input_tensor.size()

    # 展开滤波器窗口
    unfolded = F.unfold(input_tensor, kernel_size, padding=kernel_size // 2, stride=1)

    unfolded = unfolded.view(1, channels * kernel_size ** 2, -1)

    # 计算中值
    median_vals, _ = torch.median(unfolded, dim=1)

    # 重新折叠成图像
    output_tensor = F.fold(median_vals.squeeze(1), output_size=(height, width),
                           kernel_size=1, stride=1)
    # 降维
    output_tensor = output_tensor.squeeze(0)
    return output_tensor


def gaussian_kernel(kernel_size, sigma):
    """
    创建高斯核
    :param kernel_size: 核大小，应为奇数
    :param sigma: 标准差
    :return: 高斯核
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    # 创建高斯核坐标
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    y = x.unsqueeze(1).expand(-1, x.size(0))

    # 计算高斯核
    kernel = torch.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2))

    # 归一化核，使其和为1
    kernel = kernel / kernel.sum()

    # 将核扩展到适合PyTorch conv2d的形状 [out_channels, in_channels, kernel_height, kernel_width]
    return kernel.unsqueeze(0).unsqueeze(0)


def apply_gaussian_blur(image, sigma, kernel_size=5):
    """
    对灰度图像应用高斯模糊
    :param image: 输入的灰度图像张量，形状为 [1, H, W] 或 [H, W]
    :param sigma: 标准差
    :param kernel_size: 核大小，默认为5
    :return: 模糊后的图像张量
    """
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # 扩展到 [1, H, W]

    # 创建高斯核
    kernel = gaussian_kernel(kernel_size, sigma)

    # 使用 conv2d 进行卷积操作
    blurred_image = F.conv2d(image, kernel, padding=kernel_size // 2, groups=image.size(1))

    return blurred_image.squeeze(0).squeeze(0) if image.dim() == 2 else blurred_image