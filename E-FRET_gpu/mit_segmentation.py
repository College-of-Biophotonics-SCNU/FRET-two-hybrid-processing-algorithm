"""
线粒体分割
1. 由于 bax 和 bcl 聚集在线粒体上
2. 利用 AA 通道或者线粒体染色通道分割线粒体区域
3. 利用该区域进行特征提取
"""
from PIL import Image
from tranformer import custom_to_float32_tensor
from skimage.filters import threshold_otsu
from draw_plt import draw_single, save_image
import torch
import torchvision.transforms as transforms


def mit_segmentation_ostu(mit_image, mask_image, threshold_weight=1):
    # 获取唯一的细胞区域标签
    unique_labels = torch.unique(mask_image)[1:]
    region_otsu = torch.zeros_like(mit_image)
    # 对每个细胞区域进行 Otsu 阈值分割
    for label in unique_labels:
        mit_img_valid_arr = mit_image[mask_image == label]
        mit_img_valid_arr_np = mit_img_valid_arr.numpy()
        thresh = threshold_otsu(mit_img_valid_arr_np) * threshold_weight
        segmented_cell_region = torch.where(mit_img_valid_arr > thresh,
                                            torch.tensor(255, dtype=torch.float32),
                                            torch.tensor(0, dtype=torch.float32))
        region_otsu[mask_image == label] = segmented_cell_region
    return region_otsu


def save_mit_segmentation_image(mit_image_tensor, image_set_path):
    """
    保存对应的线粒体分割图像
    """
    # 原有的 float 数据换成 整数数据
    mit_image_tensor_int = mit_image_tensor.round().to(torch.int8)
    save_image(mit_image_tensor_int, image_set_path + "/mit_segmentation.tif", mode="L")


if __name__ == '__main__':
    dir_path = '../example/1_A'
    mit_image_path = dir_path + '/AA.tif'
    mask_image_path = dir_path + '/mask_img.png'

    # 转换为灰度图像
    mit_image = Image.open(mit_image_path).convert('F')
    mask_image = Image.open(mask_image_path).convert('L')
    # 定义转换操作，仅将 PIL 图像转换为 tensor，不进行归一化
    transform = transforms.Compose([
        lambda img: custom_to_float32_tensor(img)
    ])

    # 加载GPU操作
    mit_img_tensor = transform(mit_image)
    mask_img_tensor = transform(mask_image)
    result_img_tensor = mit_segmentation_ostu(mit_img_tensor, mask_img_tensor)
    save_mit_segmentation_image(result_img_tensor, dir_path)
    draw_single(result_img_tensor)

