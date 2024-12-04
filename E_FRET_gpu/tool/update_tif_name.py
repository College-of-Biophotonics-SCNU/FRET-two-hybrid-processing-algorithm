import os
"""
修改名字操作
"""
from E_FRET_gpu.tool.file_operation import get_path
from E_FRET_gpu.constant import integrity_files
def chang_BF_name(root_directory):
    """
    修改BF图像的名称
    :param root_directory:
    :return:
    """
    for root, dirs, files in os.walk(root_directory):
        image_files = [f for f in files if f.startswith("image_")]
        if len(image_files) >= 3:
            image_files.sort()
            new_names = ["BF_1.tif", "BF_2.tif", "BF_3.tif"]
            for old_name, new_name in zip(image_files, new_names):
                os.rename(os.path.join(root, old_name), os.path.join(root, new_name))
        if len(image_files) == 1:
            os.rename(os.path.join(root, image_files[0]), os.path.join(root, "BF_1.tif"))


def check_file_integrity(root_directory):
    """
    查看文件完整性 是否存在目标6个文件
    :param root_directory:
    :return:
    """
    site_sub_dirs, _ = get_path(root_directory)
    for site_sub_dir in site_sub_dirs:
        site_sub_dir_path = os.path.join(root_directory, site_sub_dir)
        for name in integrity_files:
            if not os.path.exists(os.path.join(str(site_sub_dir_path), name)):
                print(site_sub_dir_path)
                break


if __name__ == '__main__':
    check_file_integrity(r'D:\data\20240716\A549-ABT199-4h')