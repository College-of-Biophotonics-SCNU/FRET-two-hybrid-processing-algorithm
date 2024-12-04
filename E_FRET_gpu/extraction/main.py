from E_FRET_gpu.extraction.ph_local_pca_feature_extraction import LocalPCAFeatureExtraction
from E_FRET_gpu.extraction.ph_overall_pca_feature_extraction import OverallPCAFeatureExtraction

if __name__ == '__main__':
    experiment = '20240803_MCF7_BF'
    save_path = 'D:\处理结果'
    file_path = f'C://Code//python//FRET-two-hybrid-processing-algorithm//data//csv//{experiment}.csv'
    local_pca_extraction = LocalPCAFeatureExtraction(file_path, n_components=10, save_root=save_path)
    local_pca_extraction.start(need_pretreatment=False)
    overall_pca_extraction = OverallPCAFeatureExtraction(file_path, n_components=10, save_root=save_path)
    overall_pca_extraction.start(need_pretreatment=False)