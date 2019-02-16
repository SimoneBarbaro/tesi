import os
import scipy.io
from scipy import ndimage
import numpy as np


class Dataset:

    def __init__(self):
        self.images_width = 64
        self.input_shape = (self.images_width, self.images_width, 3)
        mat = scipy.io.loadmat('EILAT_data.mat')
        data = mat["data"]
        self.directory = os.path.join(data[0, 0][0][0][0], data[0, 0][0][1][0])
        self.labels = data[0, 1][0]
        self.num_classes = len(self.labels)
        for i in range(0, self.num_classes):
            self.labels[i] = self.labels[i] - 1
        self.folds = data[0, 2]
        self.dim1 = data[0, 3][0][0]
        self.dim2 = data[0, 4][0][0]
        img_paths = data[0, 5][0]
        img_names = data[0, 6][0]
        self.imgs = []
        for i, img_path in enumerate(img_paths):
            file_path = os.path.join(self.directory, img_path[0], img_names[i][0])
            image = ndimage.imread(file_path, mode="RGB")
            self.imgs.append(image)

    def get_data(self, fold, min_width=None):
        if min_width is not None and min_width > self.images_width:
            return PaddedData(self, fold, min_width)
        return Data(self, fold)


class Data:
    def __init__(self, dataset, val_fold):
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
            for i in dataset.folds[fold][dataset.dim1:dataset.dim2]:
                tmp_x.append(dataset.imgs[i - 1])
                tmp_y.append(dataset.labels[i - 1])
            for i in dataset.folds[fold][dataset.dim1:dataset.dim2]:
                tmp_x.append(dataset.imgs[i - 1])
                tmp_y.append(dataset.labels[i - 1])
        self.training_x = np.array(self.training_x)
        self.training_y = np.array(self.training_y)
        self.validation_x = np.array(self.validation_x)
        self.validation_y = np.array(self.validation_y)
        self.testing_x = np.array(self.testing_x)
        self.testing_y = np.array(self.testing_y)


class PaddedData(Data):
    def __init__(self, dataset, val_fold, new_width):
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
