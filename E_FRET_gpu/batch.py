import os
import re

import pandas as pd
from E_FRET_gpu.compute import FRETComputer
from E_FRET_gpu.extraction.ed_egfr_feature_extraction import EGFRFeatureExtraction
from E_FRET_gpu.segmentation.cell_segmentation import SegmentationModel
from E_FRET_gpu.tool.file_operation import list_immediate_subdirectories, list_numeric_subdirectories, have_FRET_target_image


def parse_batch_dir_string(input_string):
    """
    从给定的字符串中解析出 cell, treatment 和 hour 的值。
    支持小时部分为整数或浮点数。

    :param input_string: 输入字符串，格式为 'Cell-Treatment-Hour'
    :return: 包含 cell, treatment 和 hour 键值对的字典
    """
    # 定义正则表达式模式，支持浮点数
    pattern = r'^(?P<Metadata_cell>[^-]+)-(?P<Metadata_treatment>[^-]+)-(?P<Metadata_hour>\d*\.?\d+)h$'

    match = re.match(pattern, input_string)

    if match:
        result = match.groupdict()
        # 将 hour 中的 'h' 去掉并转换为浮点数
        result['Metadata_hour'] = float(result['Metadata_hour']) if result['Metadata_hour'] else None
        return result
    else:
        raise ValueError(f"输入字符串 '{input_string}' 不符合预期格式。")

class BatchProcessing:
    """
    批处理流程
    """
    def __init__(self, root):
        # 单个批次文件路径
        self.root = root
        self.batch_dir_list = []
        self.current_dir_name = os.path.basename(path)
        self.current_Metadata_cell = ''
        self.current_Metadata_site = ''
        self.current_Metadata_hour = 0
        self.current_Metadata_treatment = ''
        self.current_image_set_path = ''
        # 最后的批文件
        self.current_batch_data_df = pd.DataFrame()

    def start(self, process_function, *args, **kwargs):
        """
        开始函数
        """
        self.batch_dir_list = list_immediate_subdirectories(self.root)
        # 遍历同个实验下不同批次文件
        for batch_dir in self.batch_dir_list:
            Metadata = parse_batch_dir_string(batch_dir)
            self.current_Metadata_cell = Metadata['Metadata_cell']
            self.current_Metadata_hour = Metadata['Metadata_hour']
            self.current_Metadata_treatment = Metadata['Metadata_treatment']
            # 获取文件夹下所有的视野文件夹列表
            batch_dir_path = str(os.path.join(self.root, batch_dir))
            batch_site_dir_list = list_numeric_subdirectories(batch_dir_path)
            # 遍历批次文件夹下的不同视野文件
            for batch_site_dir in batch_site_dir_list:
                site_dir_path = str(os.path.join(batch_dir_path, batch_site_dir))

                # 验证文件完整性 确保是否存在三个通道 AA、DD、DA通道
                if not have_FRET_target_image(site_dir_path):
                    continue
                self.current_Metadata_site = int(batch_site_dir)
                # 开始业务处理
                Ed_df = process_function(site_dir_path, *args, **kwargs)
                # 保存基本信息到原有的文件上
                Ed_df['Metadata_cell'] = self.current_Metadata_cell
                Ed_df['Metadata_hour'] = self.current_Metadata_hour
                Ed_df['Metadata_treatment'] = self.current_Metadata_treatment
                Ed_df['Metadata_site'] = self.current_Metadata_site
                Ed_df.to_csv(os.path.join(site_dir_path, 'site_Ed.csv'), index=False)
                # 将 Ed_df 拼接到当前批次的数据上
                self.current_batch_data_df = pd.concat([self.current_batch_data_df, Ed_df], ignore_index=True)

            self.current_batch_data_df.to_csv(os.path.join(batch_dir_path, 'batch_Ed_features.csv'), index=False)
        # 保存最后的数据结果
        self.save_result()



    def save_result(self):
        """
        保存结果
        """
        self.current_batch_data_df.to_csv(os.path.join(self.root, 'Ed_features.csv'), index=False)


def EGFR_A549_process(image_set_path, seg_model, fret_model, batch_model):
    #############################
    # EGFR-FRET分析流程
    #############################

    # 进行分割流程
    seg_model.process(image_set_path)
    # 进行Ed效率计算流程
    fret_model.process_fret_computer(image_set_path)
    # 特征提取流程
    extraction = EGFRFeatureExtraction(image_set_path)
    Ed_pd = extraction.start()
    return Ed_pd


if __name__ == '__main__':
    path = r'D:\data\qrm\EGFR_CFP+GRB2_YFP\2024.06.11_A549_SN_8H'
    seg = SegmentationModel(diameter=200, min_box=100, max_box=400)
    fret = FRETComputer(expose_times=(200, 1000, 500))
    batch = BatchProcessing(path)
    batch.start(EGFR_A549_process, seg, fret, batch)