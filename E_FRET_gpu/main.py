"""
主要启动函数
"""
import os.path

from tool.file_operation import get_path
from segmentation.cell_segmentation import SegmentationModel
from tool.update_tif_name import chang_BF_name, check_file_integrity
root = r"D:\data\20240615"


"""
统一进行单细胞分割，用于
1. cellprofiler流程进行明场特征提取
2. FRET特征提取流程

额外流程：
1. 对于文件进行名字修改，将image_开头的名称修改为BF开头
"""
# 遍历整个实验组数据
well_sub_dirs, _ = get_path(root)
segmentationModel = SegmentationModel()
for well_sub_dir in well_sub_dirs:
     # 修改 BF 明场图像名称
     sub_dir_path = os.path.join(root, well_sub_dir)
     chang_BF_name(sub_dir_path)
     # check_file_integrity(sub_dir_path)
     # 实现单细胞分割功能
     segmentationModel.root = sub_dir_path
     segmentationModel.start()


"""
下面开始 进行FRET特征提取操作
1. 提取线粒体区域的Ed特征
2. 提取线粒体上的聚点特征

特征主要是包含
1. 效率量化特征
2. 定位量化特征
"""

# 遍历一遍子文件列表
# sub_dirs, _ = get_path(root)
# fret = FRETComputer()
# for sub_dir in sub_dirs:
#     sub_dir_path = os.path.join(root, sub_dir)
#     have_target_files = True
#     # 判断是否存在对应的文件
#     for target_file in target_files:
#         if not os.path.exists(os.path.join(sub_dir_path, target_file)):
#             have_target_files = False
#             break
#     if not have_target_files:
#         continue
#     # 开始统计 Ed 图像的单细胞情况
#     fret.process_fret_computer(sub_dir_path)
#     # 开始提取精细化单细胞特征情况
#     extraction(sub_dir_path)


