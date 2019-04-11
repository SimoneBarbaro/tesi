from data.data import CVData
from data.preprocessing import *


class DataFactory:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def build_data(self, train_index, test_index, preprocessing=None, augmentation=None, **preprocessing_args):
        result = CVData(self.dataset, train_index, test_index)
        if preprocessing == "padding":
            result = PaddedData(self.dataset, result, preprocessing_args.get("num_tiles", 1))
        elif preprocessing == "tiling":
            result = TiledData(self.dataset, result, preprocessing_args.get("num_tiles", 1))
        elif preprocessing == "hsv":
            result = HSVData(self.dataset, result)
        elif preprocessing == "rgb_lbp":
            result = RgbLBPData(self.dataset, result)  # TODO parameters
        elif preprocessing == "wavelet":
            result = WaveletData(self.dataset, result)
        if augmentation is not None:
            builder = DataAugmentationBuilder()
            for aug in augmentation:
                if aug == "feature_standardization":
                    builder.set_feature_standardization()
                elif aug == "zca_whitening":
                    builder.set_zca_whitening()
                elif aug == "rotation":
                    builder.set_rotation()
                elif aug == "shift":
                    builder.set_shift()
                elif aug == "flip":
                    builder.set_flip()
                elif aug == "zoom":
                    builder.set_zoom()
                elif aug == "fill":
                    builder.set_fill()
                elif aug == "rescale":
                    builder.set_rescale()
            result = AugmentedData(self.dataset, result, builder.build())
        return result
