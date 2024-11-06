"""
主要启动函数
"""
import os.path

from file_operation import get_path
from constant import target_files
from compute import FRETComputer
from segmentation import SegmentationModel
from feature_extraction import extraction
root = r"D:\data\20240716\A-A549-2"

# 分割操作
# segmentationModel = SegmentationModel(root=root)
# segmentationModel.start()


# 遍历一遍子文件列表
sub_dirs, _ = get_path(root)
fret = FRETComputer()
for sub_dir in sub_dirs:
    sub_dir_path = os.path.join(root, sub_dir)
    have_target_files = True
    # 判断是否存在对应的文件
    for target_file in target_files:
        if not os.path.exists(os.path.join(sub_dir_path, target_file)):
            have_target_files = False
            break
    if not have_target_files:
        continue
    # 开始统计 Ed 图像的单细胞情况
    fret.process_fret_computer(sub_dir_path)
    # 开始提取精细化单细胞特征情况
    extraction(sub_dir_path)


