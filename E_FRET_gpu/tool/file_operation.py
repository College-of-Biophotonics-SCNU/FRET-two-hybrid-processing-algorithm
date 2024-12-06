import os
import re

from E_FRET_gpu.constant import target_files
"""
文件操作文件类
"""


def get_path(main_dir_path: str):
    """
    待处理文件夹路径获取
    :param main_dir_path: 待处理文件夹路径
    :return: 文件列表，文件数量
    """
    subdir = [f for f in os.listdir(main_dir_path) if os.path.isdir(os.path.join(main_dir_path, f))]
    length = len(subdir)
    return subdir, length


def create_file(curr_subdir_path: str):
    """
    创建结果目录
    :param curr_subdir_path: 子文件夹路径
    :return: 保存处理结果路径
    """
    result_dir_path = os.path.join(curr_subdir_path, 'E-FRET_results')
    # 结果目录不存在则创建
    if not os.path.isdir(result_dir_path):
        os.makedirs(result_dir_path)
    return result_dir_path


def have_FRET_target_image(curr_subdir_path: str):
    """
    查看是否存在对应的 三通道荧光图片
    """
    for name in target_files:
        if not os.path.exists(os.path.join(curr_subdir_path, name)):
            return False
    return True

def list_immediate_subdirectories(path):
    """
    列出指定路径下一级的所有子文件夹名称。

    :param path: 要遍历的根目录路径
    :return: 子文件夹名称的列表
    """
    try:
        return [name for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))]
    except FileNotFoundError:
        print(f"指定的路径 {path} 不存在。")
        return []


def list_numeric_subdirectories(path):
    """
    列出指定路径下一级的所有名称为数字的子文件夹。

    :param path: 要遍历的根目录路径
    :return: 数字名称子文件夹的列表
    """
    numeric_pattern = re.compile(r'^\d+$')  # 匹配纯数字的正则表达式

    try:
        return [name for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name)) and numeric_pattern.match(name)]
    except FileNotFoundError:
        print(f"指定的路径 {path} 不存在。")
        return []