"""
线粒体分割
1. 由于 bax 和 bcl 聚集在线粒体上
2. 利用 AA 通道或者线粒体染色通道分割线粒体区域
3. 利用该区域进行特征提取
"""
import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage import measure, segmentation, morphology
from skimage.filters import threshold_otsu
from E_FRET_gpu.tool.draw_plt import draw_single, save_image

def mit_segmentation_ostu_single_cell(mit_image_tensor, mask_image_tensor, threshold_weight=1):
    """
    利用大津阈值法按照单细胞区域分割线粒体图像区域
    """
    # 获取唯一的细胞区域标签
    unique_labels = torch.unique(mask_image_tensor)[1:]
    region_otsu = torch.zeros_like(mit_image_tensor)
    # 对每个细胞区域进行 Otsu 阈值分割
    for label in unique_labels:
        mit_img_valid_arr = mit_image_tensor[mask_image == label]
        mit_img_valid_arr_np = mit_img_valid_arr.numpy()
        thresh = threshold_otsu(mit_img_valid_arr_np) * threshold_weight
        segmented_cell_region = torch.where(mit_img_valid_arr > thresh,
                                            torch.tensor(255, dtype=torch.float32),
                                            torch.tensor(0, dtype=torch.float32))
        region_otsu[mask_image_tensor == label] = segmented_cell_region
    return region_otsu

def mit_segmentation_watershed(mit_image_np, mask_image_np):
    """
    使用分水岭算法进行分割线粒体操作
    """
    # 去噪
    blurred = cv2.GaussianBlur(mit_image_np, (5, 5), 0)

    # Otsu阈值分割
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 距离变换
    distance = ndimage.distance_transform_edt(binary)

    # 连通域标记
    markers = measure.label(binary)

    # 应用分水岭算法
    labels_ws = segmentation.watershed(-distance, markers, mask=binary)

    # 后处理，例如去除小区域
    min_size = 100  # 设置最小区域大小
    cleaned = morphology.remove_small_objects(labels_ws, min_size=min_size)
    cleaned[cleaned > 0] = 255
    # 显示结果
    cv2.imshow('Segmented Mitochondria', cleaned.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_mit_segmentation_image(mit_image_tensor, image_set_path):
    """
    保存对应的线粒体分割图像
    """
    # 原有的 float 数据换成 整数数据
    mit_image_tensor_int = mit_image_tensor.round().to(torch.int8)
    save_image(mit_image_tensor_int, image_set_path + "/mit_segmentation.tif", mode="L")


if __name__ == '__main__':
    dir_path = '../../example/mit/mit_A133_4h'
    mit_image_path = dir_path + '/mit.tif'
    mask_image_path = dir_path + '/mask_img.png'

    # 转换为灰度图像
    mit_image = Image.open(mit_image_path).convert('F')
    mask_image = Image.open(mask_image_path).convert('L')

    # 定义转换操作，仅将 PIL 图像转换为 tensor，不进行归一化
    # transform = transforms.Compose([
    #     lambda img: custom_to_float32_tensor(img)
    # ])

    # 加载GPU操作
    # mit_img_tensor = transform(mit_image)
    # mask_img_tensor = transform(mask_image)
    # print(mit_img_tensor.shape)

    mit_image_np = np.array(mit_image)
    mit_image_np = (mit_image_np / 65535.0 * 255.0).astype(np.uint8)
    mask_image_np = np.array(mask_image, dtype=np.uint8)
    mit_segmentation_watershed(mit_image_np, mask_image_np)
    # save_mit_segmentation_image(result_img_tensor, dir_path)
    # draw_single(result_img_tensor)

