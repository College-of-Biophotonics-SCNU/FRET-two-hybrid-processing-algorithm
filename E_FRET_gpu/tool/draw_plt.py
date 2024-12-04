import os.path

import numpy as np
import seaborn as sns
from PIL import Image
from matplotlib import pyplot as plt
"""
图像绘制工具类
"""


def draw_compare(original_image_tensor, proceed_image_tensor):
    """
    绘制 plt 图像进行比对
    """
    # 如果图像是灰度图且形状为 (height, width)，添加一个通道维度变为 (height, width, 1)
    if len(original_image_tensor.shape) == 2:
        original_image_tensor = original_image_tensor.unsqueeze(2)
    if len(proceed_image_tensor.shape) == 2:
        gradient_magnitude_tensor = proceed_image_tensor.unsqueeze(2)
    # 将张量转换回 numpy 数组以便使用 matplotlib 显示
    original_image = original_image_tensor.permute(1, 2, 0).numpy()
    proceed_image = proceed_image_tensor.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(proceed_image, cmap='gray')
    plt.title("Proceed Image")

    plt.show()


def draw_single(image_tensor):
    """
    单个图像tensor输出显示plt
    需要识别输入的pytorch的tensor 还是numpy的array
    """
    image = image_tensor
    if isinstance(image_tensor, np.ndarray):
        if len(image.shape) == 2:
            image = image[np.newaxis, :, :]
        if len(image.shape) == 4:
            image = image.unsqueeze(0)
        image = np.transpose(image, (1, 2, 0))
    # 如果图像是灰度图且形状为 (height, width)，添加一个通道维度变为 (height, width, 1)
    else:
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        if len(image.shape) == 4:
            image = image.squeeze(0)
        # 转为numpy
        image = image.permute(1, 2, 0).numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.title("Image")
    plt.show()


def save_image(image_tensor, image_path, mode="F"):
    """
    保存为图像
    """
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor.squeeze(0).squeeze(0)
    img = Image.fromarray(image_tensor.numpy(), mode=mode)
    img.save(image_path)

def draw_frequency_histogram(X, y, save_path=None, save_name=''):
    """
    绘制频率直方图
    :param X: 输入的表征值
    :param y: 输入的label标签
    :param save_path: 保存路径
    :param save_name: 保存的图像名称
    """
    # 将结果和标签组合成字典
    data_dict = {}
    for result, label in zip(X, y):
        if label not in data_dict:
            data_dict[label] = []
        data_dict[label].append(result)

    # 绘制频率直方图
    plt.figure(figsize=(10, 6), dpi=150)

    for label, values in data_dict.items():
        plt.hist(values, bins=20, alpha=0.5, label=label, density=True)

    plt.title('Frequency Histogram of Results by Label')
    plt.xlabel('Result Values')
    plt.ylabel('Frequency (Density)')
    plt.legend()
    plt.grid(True)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, save_name + '_hist.jpg'), dpi=300)
    else:
        plt.show()


def draw_faceted_box_plot(df, X_name, y_name, hue_name, save_path=None, save_name=''):
    """
    绘制箱型图进行计算操作
    :param df: 输入pandas数据
    :param X_name: 顶层横坐标
    :param y_name: 输入最后公式计算的表征值
    :param hue_name: 次层横坐标
    :param save_path: 保存路径
    :param save_name: 保存的图像名称
    :return:
    """
    # 使用Seaborn绘制箱型图
    plt.figure(figsize=(10, 6), dpi=150)  # 设置图形大小
    sns.boxplot(x=X_name, y=y_name, hue=hue_name, data=df)
    plt.title('Box Plot of '+ y_name +' by '+ X_name + ' and ' + hue_name)
    plt.xlabel(X_name)
    plt.ylabel(y_name)
    plt.legend(title=hue_name)
    if save_path is not None:
        plt.savefig(os.path.join(save_path, save_name + '_box.jpg'), dpi=300)
    else:
        plt.show()