import cv2
import numpy as np
from skimage import io


def apply_mask(image, mask):
    """
    应用掩膜到图像上，保留掩膜区域的数据。

    参数:
    image (numpy.ndarray): 输入图像。
    mask (numpy.ndarray): 掩膜图像。

    返回:
    masked_image (numpy.ndarray): 掩膜应用后的图像。
    """
    return cv2.bitwise_and(image, image, mask=mask)


def detect_bright_spots(image, mask, threshold=127):
    """
    在单细胞区域内检测亮点聚点。

    参数:
    image (numpy.ndarray): 输入图像。
    mask (numpy.ndarray): 掩膜图像。
    threshold (int): 阈值分割的阈值。

    返回:
    spots (list): 亮点聚点的位置列表。
    """
    # 应用掩膜
    masked_image = apply_mask(image, mask)

    # 进行二值化处理
    _, binary_image = cv2.threshold(masked_image, threshold, 255, cv2.THRESH_BINARY)

    # 确保二值化图像是uint8类型
    binary_image = binary_image.astype(np.uint8)

    # 寻找连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image)

    # 过滤出亮点聚点
    spots = []
    for stat in stats[1:]:
        # 这里可以调整大小阈值来排除小的噪点
        if stat[4] > 5:  # 聚点最小像素数
            center_x = stat[0] + stat[2] / 2
            center_y = stat[1] + stat[3] / 2
            spots.append((center_x, center_y))

    return spots


def overlay_spots_on_image(image, spots):
    """
    在图像上叠加亮点聚点的标记。

    参数:
    image (numpy.ndarray): 输入图像。
    spots (list): 亮点聚点的位置列表。

    返回:
    annotated_image (numpy.ndarray): 标记后的图像。
    """
    # 确保图像为三通道
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    annotated_image = image.copy()
    for spot in spots:
        # 画圆标记亮点聚点
        cv2.circle(annotated_image, (int(spot[0]), int(spot[1])), 5, (0, 0, 255), -1)

    return annotated_image


def main(image_path, mask_path, output_path):
    """
    主函数，加载图像和掩膜，检测亮点聚点，并保存结果图像。

    参数:
    image_path (str): 输入图像路径。
    mask_path (str): 掩膜图像路径。
    output_path (str): 输出图像路径。
    """
    # 加载图像和掩膜
    image = io.imread(image_path)
    mask = io.imread(mask_path)

    # 如果掩膜是三通道，则转换为单通道
    if len(mask.shape) == 3 and mask.shape[-1] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 检测亮点聚点
    spots = detect_bright_spots(image, mask)

    # 在图像上叠加亮点聚点
    annotated_image = overlay_spots_on_image(image, spots)

    # 保存结果图像
    save_image(output_path, annotated_image)


def save_image(output_path, image):
    """
    Save an image to a file after ensuring it is in the correct format.

    Parameters:
    output_path (str): Path to save the image.
    image (numpy.ndarray): The image array to save.
    """
    # 确保图像数据类型是uint8
    if image.dtype == np.float32 or image.dtype == np.float64:
        # 将浮点数转换为uint8
        image = (image * 255).clip(0, 255).astype(np.uint8)

    # 使用skimage保存图像
    io.imsave(output_path, image)


# 示例调用
image_path = '../example/normal/1_A/Ed.tif'
mask_path = '../example/normal/1_A/mask_img.png'
output_path = 'output/output_image.png'

main(image_path, mask_path, output_path)