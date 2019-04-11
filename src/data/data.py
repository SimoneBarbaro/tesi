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
