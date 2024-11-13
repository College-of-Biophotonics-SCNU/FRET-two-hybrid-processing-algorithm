import os
"""
修改名字操作
"""

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