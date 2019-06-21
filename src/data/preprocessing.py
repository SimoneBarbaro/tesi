import random

from data.dataset import Dataset
from data.data import Data
import matplotlib
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import pywt
from skimage.feature import local_binary_pattern
from skimage import color


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


class ChangedImagesData(PreprocessedData):
    def __init__(self, dataset: Dataset, data: Data):
        super(ChangedImagesData, self).__init__(dataset, data)
        self.training_x = np.array(list(map(self._change_image, self.training_x)))
        self.validation_x = np.array(list(map(self._change_image, self.validation_x)))
        self.testing_x = np.array(list(map(self._change_image, self.testing_x)))

    def _change_image(self, image):
        raise NotImplementedError


class HSVData(ChangedImagesData):
    def _change_image(self, image):
        return color.rgb2hsv(image)


class LABData(ChangedImagesData):
    def _change_image(self, image):
        return color.rgb2lab(image)


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


class FullWaveletData(ChangedImagesData):

    @staticmethod
    def __make_grey_wavelet(image):
        approximation, (LH, HL, HH) = pywt.dwt2(image, "db1")
        r1 = np.concatenate((approximation, LH), axis=0)
        r2 = np.concatenate((HL, HH), axis=0)
        result = np.concatenate((r1, r2), axis=1)
        return result

    def _change_image(self, image):
        b, g, r = cv2.split(image)
        b1 = FullWaveletData.__make_grey_wavelet(b)
        g1 = FullWaveletData.__make_grey_wavelet(g)
        r1 = FullWaveletData.__make_grey_wavelet(r)

        return cv2.merge([b1, g1, r1])


class AugmentedData(PreprocessedData):
    def __init__(self, dataset: Dataset, data: Data, generator):
        super(AugmentedData, self).__init__(dataset, data)
        self.generator = generator

    def get_generator(self):
        return self.generator


class TileGenerator(keras.utils.Sequence):
    def __init__(self, input_shape, batch_size, data_lenght, tile_size):
        self.__len = data_lenght // batch_size
        self.__tile_size = tile_size
        self.__img_shape = [input_shape[0], input_shape[1]]
        self.__i = 0
        self.__offsets = []
        for i in range(self.__len):
            self.__offsets.append((random.randint(0, self.__img_shape[0] - tile_size),
                                   random.randint(0, self.__img_shape[1] - tile_size)))

    def __len__(self):
        return self.__len

    def __getitem__(self, item):
        result = item[self.__offsets[1] * self.__i:self.__offsets[1] * self.__i + self.__tile_size,
                      self.__offsets[0] * self.__i:self.__offsets[0] * self.__i + self.__tile_size]
        self.__i = self.__i + 1
        return result


class SubTiledData(AugmentedData):

    def __init__(self, dataset: Dataset, data: Data, batch_size, tile_size=32):
        super(SubTiledData, self).__init__(dataset, data,
                                           TileGenerator(data.input_shape, batch_size, len(data.training_x), tile_size))


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
