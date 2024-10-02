import os
from constant import target_files
"""
文件操作文件类
"""


def get_path(main_dir_path: str):
    """
    待处理文件夹路径获取
    :param main_dir_path: 待处理文件夹路径
    :return: 文件列表，文件数量
    """
    subdir = os.listdir(main_dir_path)
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


if __name__ == "__main__":
    test_dir = r'C:\Users\22806\Downloads\0'
    print(have_FRET_target_image(test_dir))
