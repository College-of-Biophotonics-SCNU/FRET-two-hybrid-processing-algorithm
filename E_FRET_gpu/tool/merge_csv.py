"""
合并 excel 统一为一个excel进行输出统计计算
需要添加的参数遍历
1. Metadata_site
"""
import os
import pandas as pd
from file_operation import get_path


def merge_Ed_feature_csv(root=None):
    merged_df = pd.DataFrame()
    # 遍历一遍子文件列表
    sub_dirs, _ = get_path(root)
    for sub_dir in sub_dirs:
        sub_dir_path = str(os.path.join(root, sub_dir))

        for file_name in os.listdir(sub_dir_path):
            if file_name == 'cell_Ed_averages.csv':
                file_path = os.path.join(sub_dir_path, file_name)
                df = pd.read_csv(file_path)
                df['site'] = sub_dir
                merged_df = pd.concat([merged_df, df], ignore_index=True)
    merged_df.rename(columns={'index': 'ObjectNumber'}, inplace=True)
    merged_df.to_csv(os.path.join(root, 'Ed.csv'), index=False)


if __name__ == '__main__':
    file_root = r"D:\data\20240716\A-A549-2"
    merge_Ed_feature_csv(file_root)