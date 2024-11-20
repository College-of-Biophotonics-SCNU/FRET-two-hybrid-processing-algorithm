import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

image_path = "../example/normal/1_A/Ed.tif"
mask_path = "../example/normal/1_A/mask_img.png"
aggregate = "../example/1_A/aggregates.tif"
# 将聚点和灰度图合并
# 读取图像
aggregate_image = Image.open(aggregate).convert('F')
aggregate_image_np = np.array(aggregate_image)
image = Image.open(image_path).convert('F')
image_np = np.array(image)

# 将 image_np 乘以 255 并转换为整数类型
image_np = (image_np * 255).astype(np.uint8)
print(aggregate_image_np.shape)
# 创建一个 RGB 版本的图像用于输出，初始化为输入的灰度图像转换后的 RGB 形式
output_image = np.stack((image_np, image_np, image_np), axis=-1)

# 将 aggregate 中值为 1 的点对应在输出图像上的位置设置为红色
red_color = [255, 0, 0]
for i in range(aggregate_image_np.shape[0]):
    for j in range(aggregate_image_np.shape[1]):
        if aggregate_image_np[i, j] == 1:
            output_image[i, j] = red_color

# 显示输出图像
plt.imshow(output_image)
plt.show()