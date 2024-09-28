import torch
from PIL import Image



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
                 BACKGROUND_THRESHOLD: float = 1.2,
                 pcolor: bool = False,
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
        self.pcolor = pcolor
        self.subdir_list = []  # 子目录列表
        self.subdir_len = 0  # 子目录长度
        self.main_dir = main_dir
        self.gpu = gpu
        self.batch_size = batch_size
        self.have_gpu()

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


if __name__ == "__main__":
    fret = FRETComputer()
    fret.have_gpu()

