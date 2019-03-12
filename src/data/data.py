from data.dataset import Dataset
import numpy as np


class Data:
    def __init__(self, dataset: Dataset):
        self.input_shape = dataset.input_shape
        self.num_classes = dataset.num_classes
        self.training_x = []
        self.training_y = []
        self.validation_x = []
        self.validation_y = []
        self.testing_x = []
        self.testing_y = []

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
        for fold in range(len(dataset.folds)):
            self._split_train_test(dataset, fold)
        self._wrap_data()
        self.validation_x = self.training_x[val_index]
        self.validation_y = self.training_y[val_index]
        self.training_x = self.training_x[train_index]
        self.training_y = self.training_y[train_index]


class PaddedData(CVData):
    def __init__(self, dataset: Dataset, val_fold, new_width):
        super(PaddedData, self).__init__(dataset, val_fold)
        self.input_shape = (new_width, new_width, self.input_shape[2])
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


class TiledData(CVData):
    def __init__(self, dataset: Dataset, val_fold, num_tiles):
        super(TiledData, self).__init__(dataset, val_fold)
        self.input_shape = (self.input_shape[0] * num_tiles, self.input_shape[1] * num_tiles, self.input_shape[2])
        self.training_x = np.tile(self.training_x, (1, num_tiles, num_tiles, 1))
        self.validation_x = np.tile(self.validation_x, (1, num_tiles, num_tiles, 1))
        self.testing_x = np.tile(self.testing_x, (1, num_tiles, num_tiles, 1))


class DataFactory:
    def __init__(self, datasset: Dataset):
        self.dataset = datasset

    def build_data(self, validation_fold=0, train_index=None, test_index=None, preprocessing=None, **preprocessing_args):
        # TODO the rest
        if train_index is None or test_index is None:
            return OneFoldData(self.dataset, validation_fold)
        return SklearnCVData(self.dataset, train_index, test_index)
