import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def are_images_equal(image_path1, image_path2):
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    # 将图像转换为numpy数组
    img1_numpy = np.array(img1)
    img2_numpy = np.array(img2)

    if img1.size != img2.size:
        return False

    img1_numpy[img1_numpy > 0] = 1
    img2_numpy[img2_numpy > 0] = 1

    print(img1_numpy[img1_numpy > 0].size)
    print(img2_numpy[img2_numpy > 0].size)
    # 找出不重叠的部分
    non_overlap = np.where((img1_numpy != img2_numpy), img1_numpy, 0)
    # 显示图像
    plt.imshow(non_overlap)
    plt.show()


image_path1 = r'C:\Code\python\FRET-two-hybrid-processing-algorithm\verification_tool\output\gpu_Ed.tif'
image_path2 = r'C:\Code\python\FRET-two-hybrid-processing-algorithm\verification_tool\output\matlab_Ed.tif'

if are_images_equal(image_path1, image_path2):
    print("两张图片像素值相等。")
else:
    print("两张图片像素值不相等。")
