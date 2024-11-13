import os.path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tifffile import tifffile
from constant import target_files, mask_filename
"""
FRET 效率计算函数
注意数据加载顺序是 AA、DD、DA
"""


def load_image_to_tensor(image_path):
    """
    加载图像到 GPU 上
    """
    img = Image.open(image_path)
    return torch.from_numpy(np.array(img)).unsqueeze(0).float()


class FRETComputer:
    """
    E-FRET 计算
    这套参数测量于2023年9月28日
    """

    def __init__(self,
                 a: float = 0.150332,
                 b: float = 0.001107,
                 c: float = 0.000561,
                 d: float = 0.780016,
                 G: float = 5.494216,
                 k: float = 0.432334,
                 expose_times: tuple = (300, 300, 300),
                 BACKGROUND_THRESHOLD: float = 1.0,
                 is_pcolor: bool = False,
                 main_dir: str = '',
                 gpu: bool = True,
                 batch_size: int = 10
                 ):
        """
        :param a: 校正因子a
        :param b: 校正因子b
        :param c: 校正因子c
        :param d: 校正因子d
        :param G: 校正因子G
        :param k: 校正因子k
        :param BACKGROUND_THRESHOLD: 背景模板波长阈值超参数
        :param expose_times: 三通道曝光时间
        :param pcolor: 是否开启伪彩图显示
        :param main_dir: 文件存在路径
        :param gpu: 是否开启 gpu 计算
        :param batch_size: gpu 处理过程中存放图像的批次大小
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.G = G
        self.k = k
        self.expose_times = expose_times
        self.BACKGROUND_THRESHOLD = BACKGROUND_THRESHOLD
        self.is_pcolor = is_pcolor
        self.subdir_list = []                       # 子目录列表
        self.subdir_len = 0                         # 子目录长度
        self.current_sub_path = ''                  # 当前处理的子文件夹
        self.main_dir = main_dir
        self.gpu = gpu
        self.batch_size = batch_size
        self.device = self.have_gpu()

    @staticmethod
    def have_gpu():
        # 检查是否有可用的 GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("启用设备为: ", device)
        return device

    def process_folder_batch(self):
        """
        批处理流程，加载图像进行并行运算操作
        """
        pass

    def process_fret_computer(self, sub_path):
        """
        利用 pytorch 进行 fret 效率计算操作
        """
        self.current_sub_path = sub_path
        # 加载图像为 tensor 并移动到相同设备（假设都在 CPU 或都在 GPU）
        image_AA = load_image_to_tensor(os.path.join(sub_path, target_files[0]))
        image_DD = load_image_to_tensor(os.path.join(sub_path, target_files[1]))
        image_DA = load_image_to_tensor(os.path.join(sub_path, target_files[2]))
        image_mask = load_image_to_tensor(os.path.join(sub_path, mask_filename))

        # image_mask 存在单细胞信息值 需要只包含0和1位置信息情况
        mask = image_mask.clone()
        mask[mask > 0] = 1
        # 计算背景噪声 并且FRET三通道减去对应的背景噪声 添加 mask 屏蔽
        image_AA, image_AA_template = self.subtract_background_noise(image_AA, mask)
        image_DD, image_DD_template = self.subtract_background_noise(image_DD, mask)
        image_DA, image_DA_template = self.subtract_background_noise(image_DA, mask)
        # 添加三通道有效模板 三通道值全部必须为正
        effective_template = image_AA_template * image_DD_template * image_DA_template * mask
        # 计算 Fc 图像
        Fc = image_DA - self.a * (image_AA - self.c * image_DD) - self.d * (image_DD - self.b * image_AA)
        Fc[Fc < 0] = 0
        print(sub_path, " this set FRET images 有效Fc最小值为", Fc[Fc > 0].min())
        # 计算 Ed 效率以及 Rc 浓度值
        Ed = Fc / (Fc + self.G * image_DD + 1e-7) * effective_template
        # Rc = (self.k * image_AA) / (image_DD + Fc / self.G)
        # 保存Ed效率图 保存为 TIFF 文件（可以选择其他格式，如 PNG），并设置保存参数以保留浮点数精度
        tifffile.imwrite(os.path.join(self.current_sub_path, 'Ed.tif'), Ed.squeeze(0).numpy())
        # 统计单细胞效率情况
        self.count_single_cell_Ed(Ed, image_mask)
        # 绘制伪彩图
        self.draw_color_map(Ed.squeeze(0).numpy(), "Ed")

    def count_single_cell_Ed(self, image, mask):
        """
        1. 计算单细胞的 Ed 效率值，并统计所有的效率分布情况
        """
        cell_averages = {}
        # 统计单个细胞中的 Ed 平均效率情况
        for cell_id in range(1, int(mask.max().item()) + 1):
            cell_mask = (mask == cell_id)
            cell_intensities = image[cell_mask]
            if len(cell_intensities) > 0:
                average_intensity = cell_intensities.mean().item()
                cell_averages[cell_id] = {}
                cell_averages[cell_id]['fret_cell_Ed_averages'] = average_intensity
                cell_averages[cell_id]['fret_cell_Ed_variance'] = (cell_intensities - average_intensity).pow(2).mean().item()

        # 创建一个 DataFrame
        df = pd.DataFrame.from_dict(cell_averages, orient='index')

        # 将 DataFrame 保存为 CSV 文件
        df.to_csv(os.path.join(self.current_sub_path, 'cell_Ed_averages.csv'), index_label='index', index=True)

    def draw_color_map(self, image_np, hist_name):
        """
        绘制伪彩图图像
        """
        # 创建自定义颜色映射
        cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 'blue'), (0.5, 'cyan'), (1, 'yellow')])
        # 使用 imshow 绘制伪彩图
        print(self.current_sub_path, " this set FRET images 最大值 ",
              hist_name, " 为 ", np.max(image_np), "最小值 ",
              hist_name, " 为 ", np.min(image_np))
        plt.imshow(image_np, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(os.path.join(self.current_sub_path, hist_name + "wei.jpg"))
        # 清除所有plt的参数
        plt.clf()

    def subtract_background_noise(self, image, mask, only_background=True, background_threshold=1.2):
        """
        1. 利用直方图统计法 计算出对应的背景噪声 同时将图像减去背景噪声
        2. 掩码对应的图像
        """
        background_flat = image.squeeze().flatten().numpy()
        if only_background:
            # 反转 mask
            inverted_mask_tensor = 1 - mask
            # 用反转后的掩码与 AA 图像相乘得到背景区域
            background_tensor = image * inverted_mask_tensor
            # 将背景tensor转换为一维数组以便进行直方图统计
            background_flat = background_tensor.squeeze().flatten().numpy()
        # 统计直方图
        hist, bin_edges = np.histogram(background_flat, bins=np.arange(1, 2001, 1))
        # 找到频率最高像素值对应的索引
        most_frequent_index = np.argmax(hist)
        # 获取频率最高像素值作为背景噪声
        background_noise = bin_edges[most_frequent_index] * background_threshold
        print(self.current_sub_path,
              " 图像的最大值为", image.max(),
              " 图像最小值为", image.min(),
              " 图像的背景阈值为", bin_edges[most_frequent_index])
        # 将 AA 图像减去背景噪声 掩码屏蔽
        noise_removed_tensor = image - bin_edges[most_frequent_index]
        noise_removed_tensor[noise_removed_tensor < 0] = 0
        noise_removed_tensor[noise_removed_tensor > 50000] = 0
        # 计算该图像的有效区域的模板
        template_tensor = image - background_noise
        template_tensor[template_tensor > 0] = 1
        template_tensor[template_tensor <= 0] = 0
        # 除以曝光时间
        noise_removed_tensor = noise_removed_tensor / self.expose_times[0]
        print(self.current_sub_path,
              " 降噪图像的最大值为", noise_removed_tensor.max(),
              " 降噪图像最小值为", noise_removed_tensor[noise_removed_tensor > 0].min())
        # 返回降噪结果
        return noise_removed_tensor, template_tensor


if __name__ == "__main__":
    fret = FRETComputer()
    fret.process_fret_computer(r'../example/3_C')
