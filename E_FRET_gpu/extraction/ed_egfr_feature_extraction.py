from efficiency_feature_extraction import EFeatureExtraction
from E_FRET_gpu.segmentation.aggregates_segmentation import threshold_calculate_otsu, threshold_calculate_otsu_with_single_cell_region
from E_FRET_gpu.tool.draw_plt import draw_single


class EGFRFeatureExtraction(EFeatureExtraction):
    """
    EGFR 靶点特征提取算法
    1. 需要利用 DD 通道算法进行计算聚点定位信息
    2. 基于 DD 的共定位信息获取 Ed 图像的周围效率信息
    """
    def __init__(self, path):
        super().__init__(path)

    def start(self, threshold_weight=1.2):
        self.target_efficiency_extraction(threshold_weight)

    def target_efficiency_extraction(self, threshold_weight):
        """
        EGFR 靶点效率提取
        """
        image_agg = threshold_calculate_otsu_with_single_cell_region(self.image_DD, self.image_mask, threshold_weight=threshold_weight)
        draw_single(image_agg)

if __name__ == '__main__':
    image_path = r'C:\Code\python\FRET-two-hybrid-processing-algorithm\example\egfr\control-3'
    model = EGFRFeatureExtraction(image_path)
    model.start()