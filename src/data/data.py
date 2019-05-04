from data.dataset import Dataset
import numpy as np


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


class FullData(Data):
    def __init__(self, dataset: Dataset, train_index, val_index):
        super(FullData, self).__init__(dataset)
        for i in dataset.data_indexes[0:dataset.test_data_len]:
            self.training_x.append(dataset.imgs[i - 1])
            self.training_y.append(dataset.imgs_labels[i - 1])
        self._wrap_data()
        self.validation_x = self.training_x[val_index]
        self.validation_y = self.training_y[val_index]
        self.training_x = self.training_x[train_index]
        self.training_y = self.training_y[train_index]
        self.testing_x = self.validation_x
        self.testing_y = self.validation_y


class CVData(Data):
    def __init__(self, dataset: Dataset, train_index, val_index):
        super(CVData, self).__init__(dataset)
        for i in dataset.data_indexes[0:dataset.training_date_len]:
            self.training_x.append(dataset.imgs[i - 1])
            self.training_y.append(dataset.imgs_labels[i - 1])
        for i in dataset.data_indexes[dataset.training_date_len:dataset.test_data_len]:
            self.testing_x.append(dataset.imgs[i - 1])
            self.testing_y.append(dataset.imgs_labels[i - 1])
        self._wrap_data()
        self.validation_x = self.training_x[val_index]
        self.validation_y = self.training_y[val_index]
        self.training_x = self.training_x[train_index]
        self.training_y = self.training_y[train_index]
