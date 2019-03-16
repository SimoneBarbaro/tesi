from data.dataset import Dataset
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Data:
    def __init__(self, dataset: Dataset):
        # self.input_shape = dataset.input_shape
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

    def _split_train_test(self, dataset, fold):
        for i in dataset.folds[fold][0:dataset.dim1]:
            self.training_x.append(dataset.imgs[i - 1])
            self.training_y.append(dataset.labels[i - 1])
        for i in dataset.folds[fold][dataset.dim1:dataset.dim2]:
            self.testing_x.append(dataset.imgs[i - 1])
            self.testing_y.append(dataset.labels[i - 1])


class OneFoldData(Data):
    def __init__(self, dataset: Dataset, val_fold):
        super(OneFoldData, self).__init__(dataset)

        self._split_train_test(dataset, val_fold)
        self.validation_x = self.testing_x
        self.validation_y = self.testing_y
        self._wrap_data()


class CVData(Data):
    def __init__(self, dataset: Dataset, val_fold):
        super(CVData, self).__init__(dataset)

        for fold in range(0, len(dataset.folds)):
            tmp_x = self.training_x
            tmp_y = self.training_y
            if fold == val_fold:
                tmp_x = self.validation_x
                tmp_y = self.validation_y
            for i in dataset.folds[fold][0:dataset.dim1]:
                tmp_x.append(dataset.imgs[i - 1])
                tmp_y.append(dataset.labels[i - 1])
            for i in dataset.folds[fold][dataset.dim1:dataset.dim2]:
                self.testing_x.append(dataset.imgs[i - 1])
                self.testing_y.append(dataset.labels[i - 1])
        self._wrap_data()


class SklearnCVData(Data):
    def __init__(self, dataset: Dataset, train_index, val_index):
        super(SklearnCVData, self).__init__(dataset)
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


class AugmentedData(PreprocessedData):
    def __init__(self, dataset: Dataset, data: Data, generator: ImageDataGenerator):
        super(AugmentedData, self).__init__(dataset, data)
        self.generator = generator

    def get_generator(self)-> ImageDataGenerator:
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
        return  self.__generator


class DataFactory:
    def __init__(self, datasset: Dataset):
        self.dataset = datasset

    def build_data(self, validation_fold=0, train_index=None, test_index=None,
                   preprocessing=None, augmentation=None, **preprocessing_args):
        if train_index is None or test_index is None:
            result = OneFoldData(self.dataset, validation_fold)
        else:
            result = SklearnCVData(self.dataset, train_index, test_index)
        if preprocessing == "padding":
            result = PaddedData(self.dataset, result, preprocessing_args.get("num_tiles", 1))
        elif preprocessing == "tiling":
            result = TiledData(self.dataset, result, preprocessing_args.get("num_tiles", 1))
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
