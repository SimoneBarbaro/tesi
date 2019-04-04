from data.dataset import Dataset
import matplotlib
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import pywt
from skimage.feature import local_binary_pattern


class Data:
    def __init__(self, dataset: Dataset):
        self.num_classes = dataset.num_classes
        self.training_x = []
        self.training_y = []
        self.validation_x = []
        self.validation_y = []
        self.testing_x = []
        self.testing_y = []

    @property
    def input_shape(self):
        return self.training_x.shape[1:]

    def _wrap_data(self):
        self.training_x = np.array(self.training_x)
        self.training_y = np.array(self.training_y)
        self.validation_x = np.array(self.validation_x)
        self.validation_y = np.array(self.validation_y)
        self.testing_x = np.array(self.testing_x)
        self.testing_y = np.array(self.testing_y)

    # TODO dataset fold should be changed
    def _split_train_test(self, dataset, fold):
        for i in dataset.folds[fold][0:dataset.dim1]:
            self.training_x.append(dataset.imgs[i - 1])
            self.training_y.append(dataset.labels[i - 1])
        for i in dataset.folds[fold][dataset.dim1:dataset.dim2]:
            self.testing_x.append(dataset.imgs[i - 1])
            self.testing_y.append(dataset.labels[i - 1])


class CVData(Data):
    def __init__(self, dataset: Dataset, train_index, val_index):
        super(CVData, self).__init__(dataset)
        self._split_train_test(dataset, 0)
        self._wrap_data()
        self.validation_x = self.training_x[val_index]
        self.validation_y = self.training_y[val_index]
        self.training_x = self.training_x[train_index]
        self.training_y = self.training_y[train_index]


class PreprocessedData(Data):
    def __init__(self, dataset: Dataset, data: Data):
        super(PreprocessedData, self).__init__(dataset)
        self.training_x = data.training_x
        self.training_y = data.training_y
        self.validation_x = data.validation_x
        self.validation_y = data.validation_y
        self.testing_x = data.testing_x
        self.testing_y = data.testing_y


class PaddedData(PreprocessedData):
    def __init__(self, dataset: Dataset, data: Data, num_tiles=1):
        super(PaddedData, self).__init__(dataset, data)

        new_width = self.input_shape[0] + num_tiles
        # self.input_shape = (new_width, new_width, self.input_shape[2])
        padding_l = int((new_width - dataset.images_width) / 2)
        padding_r = int((new_width - dataset.images_width + 1) / 2)
        self.training_x = np.pad(self.training_x,
                                 ((0, 0), (padding_l, padding_r), (padding_l, padding_r), (0, 0)),
                                 'constant', constant_values=[0])
        self.validation_x = np.pad(self.validation_x,
                                   ((0, 0), (padding_l, padding_r), (padding_l, padding_r), (0, 0)),
                                   'constant', constant_values=[0])
        self.testing_x = np.pad(self.testing_x,
                                ((0, 0), (padding_l, padding_r), (padding_l, padding_r), (0, 0)),
                                'constant', constant_values=[0])


class TiledData(PreprocessedData):
    def __init__(self, dataset: Dataset, data: Data, num_tiles=1):
        super(TiledData, self).__init__(dataset, data)
        # self.input_shape = (self.input_shape[0] * num_tiles, self.input_shape[1] * num_tiles, self.input_shape[2])
        self.training_x = np.tile(self.training_x, (1, num_tiles, num_tiles, 1))
        self.validation_x = np.tile(self.validation_x, (1, num_tiles, num_tiles, 1))
        self.testing_x = np.tile(self.testing_x, (1, num_tiles, num_tiles, 1))


class HSVData(PreprocessedData):
    def __init__(self, dataset: Dataset, data: Data):
        super(HSVData, self).__init__(dataset, data)
        self.training_x = self.__change_color(self.training_x)
        self.validation_x = self.__change_color(self.validation_x)
        self.testing_x = self.__change_color(self.testing_x)

    @staticmethod
    def __change_color(x):
        result = np.array(list(map(lambda p: p / 255.0, x)))
        return matplotlib.colors.rgb_to_hsv(result)


class ChangedImagesData(PreprocessedData):
    def __init__(self, dataset: Dataset, data: Data):
        super(ChangedImagesData, self).__init__(dataset, data)
        self.training_x = np.array(list(map(self._change_image, self.training_x)))
        self.validation_x = np.array(list(map(self._change_image, self.validation_x)))
        self.testing_x = np.array(list(map(self._change_image, self.testing_x)))

    def _change_image(self, image):
        raise NotImplementedError


class RgbLBPData(ChangedImagesData):
    def __init__(self, dataset: Dataset, data: Data, n_points=24, radius=3):
        self.__n_points = n_points
        self.__radius = radius
        super(RgbLBPData, self).__init__(dataset, data)

    def _change_image(self, image):
        b, g, r = cv2.split(image)
        b1 = local_binary_pattern(b, self.__n_points, self.__radius, method='uniform')
        g1 = local_binary_pattern(g, self.__n_points, self.__radius, method='uniform')
        r1 = local_binary_pattern(r, self.__n_points, self.__radius, method='uniform')
        return cv2.merge([b1, g1, r1])


class WaveletData(ChangedImagesData):

    def _change_image(self, image):
        b, g, r = cv2.split(image)
        _, b1 = pywt.dwt(b, "db1")
        _, g1 = pywt.dwt(g, "db1")
        _, r1 = pywt.dwt(r, "db1")
        return cv2.merge([b1, g1, r1])


class AugmentedData(PreprocessedData):
    def __init__(self, dataset: Dataset, data: Data, generator: ImageDataGenerator):
        super(AugmentedData, self).__init__(dataset, data)
        self.generator = generator

    def get_generator(self) -> ImageDataGenerator:
        return self.generator


class DataAugmentationBuilder:
    def __init__(self):
        self.__generator = ImageDataGenerator()

    def set_feature_standardization(self):
        self.__generator.featurewise_center = True
        self.__generator.featurewise_std_normalization = True

    def set_zca_whitening(self):
        self.__generator.zca_whitening = True

    def set_rotation(self, rotation=90):
        self.__generator.rotation_range = rotation

    def set_shift(self, shift=0.2):
        self.__generator.width_shift_range = shift
        self.__generator.height_shift_range = shift

    def set_flip(self, horizontal=True, vertical=True):
        self.__generator.horizontal_flip = horizontal
        self.__generator.vertical_flip = vertical

    def set_zoom(self, zoom=0.2):
        self.__generator.zoom_range = zoom

    def set_fill(self, fill_mode="reflect"):
        self.__generator.fill_mode = fill_mode

    def set_rescale(self, rescale=2):
        self.__generator.rescale = rescale

    def build(self):
        return self.__generator


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
