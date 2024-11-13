import os


def get_path(
        main_dir_path: str,
):
    """
    待处理文件夹路径获取
    :param main_dir_path: 待处理文件夹路径
    :return: 文件列表，文件数量
    """
    subdir = os.listdir(main_dir_path)
    length = len(subdir)
    return subdir, length


def create_file(
        curr_subdir_path: str,
):
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


def write_data(
        result_dir_path,
        ave,
        esd,
        avr,
        rsd):
    """
    将均值标准差数据写入txt文件
    :param result_dir_path:文件路径
    :param ave: 效率均值
    :param esd:效率标准差
    :param avr: 浓度比均值
    :param rsd: 浓度比标准差
    """
    calculate_dir = os.path.join(result_dir_path, 'calculateResult.txt')
    file = open(calculate_dir, 'w')
    file.write('Ed平均值(avE):' + str(ave))
    file.write('\nEd标准差(Esd):' + str(esd))
    file.write('\n浓度比平均值(avR):' + str(avr))
    file.write('\n浓度比标准差(Rsd):' + str(rsd))
    file.close()


