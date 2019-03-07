from data.dataset import Dataset
import numpy as np


class Data:
    def __init__(self, dataset: Dataset, val_fold):
        self.input_shape = dataset.input_shape
        self.num_classes = dataset.num_classes
        self.training_x = []
        self.training_y = []
        self.validation_x = []
        self.validation_y = []
        self.testing_x = []
        self.testing_y = []
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
        self.training_x = np.array(self.training_x)
        self.training_y = np.array(self.training_y)
        self.validation_x = np.array(self.validation_x)
        self.validation_y = np.array(self.validation_y)
        self.testing_x = np.array(self.testing_x)
        self.testing_y = np.array(self.testing_y)


class PaddedData(Data):
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
        self.testing_x = np.pad(self.validation_x,
                                ((0, 0), (padding_l, padding_r), (padding_l, padding_r), (0, 0)),
                                'constant', constant_values=[0])


class TiledData(Data):
    def __init__(self, dataset: Dataset, val_fold, num_tiles):
        super(TiledData, self).__init__(dataset, val_fold)
        self.input_shape = (self.input_shape[0] * num_tiles, self.input_shape[1] * num_tiles, self.input_shape[2])
        self.training_x = np.tile(self.training_x, (1, num_tiles, num_tiles, 1))
        self.validation_x = np.tile(self.validation_x, (1, num_tiles, num_tiles, 1))
        self.testing_x = np.tile(self.testing_x, (1, num_tiles, num_tiles, 1))


class DataFactory:
    def __init__(self, datasset: Dataset):
        self.dataset = datasset

    def build_data(self, validation_fold=0, preprocessing=None, **preprocessing_args):
        # TODO the rest
        return Data(self.dataset, validation_fold)
