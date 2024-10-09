from PIL import Image
from matplotlib import pyplot as plt
"""
图像绘制工具类
"""


def draw_compare(original_image_tensor, gradient_magnitude_tensor):
    """
    绘制 plt 图像进行比对
    """
    # 如果图像是灰度图且形状为 (height, width)，添加一个通道维度变为 (height, width, 1)
    if len(original_image_tensor.shape) == 2:
        original_image_tensor = original_image_tensor.unsqueeze(2)
    if len(gradient_magnitude_tensor.shape) == 2:
        gradient_magnitude_tensor = gradient_magnitude_tensor.unsqueeze(2)
    # 将张量转换回 numpy 数组以便使用 matplotlib 显示
    original_image = original_image_tensor.permute(1, 2, 0).numpy()
    gradient_image = gradient_magnitude_tensor.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(gradient_image, cmap='gray')
    plt.title("Gradient Magnitude (Contour)")

    plt.show()


def draw_single(image_tensor):
    """
    单个图像tensor输出显示plt
    """
    # 如果图像是灰度图且形状为 (height, width)，添加一个通道维度变为 (height, width, 1)
    if len(image_tensor.shape) == 2:
        image_tensor = image_tensor.unsqueeze(2)
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor.squeeze(0)
    # 转为numpy
    image = image_tensor.permute(1, 2, 0).numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.title("Image")
    plt.show()


def save_image(image_tensor, image_path):
    """
    保存为图像
    """
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor.squeeze(0).squeeze(0)
    img = Image.fromarray(image_tensor.numpy(), mode='F')
    img.save(image_path)

