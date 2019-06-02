import os
import scipy.io
import numpy as np
import matplotlib


class Dataset:

    def __init__(self, mat_file):
        mat = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, mat_file))
        data = mat["data"]
        self.__directory = os.path.join(data[0, 0][0][0][0], data[0, 0][0][1][0])
        self.imgs_labels = data[0, 1][0]
        self.num_classes = max(self.imgs_labels)
        for i in range(0, len(self.imgs_labels)):
            self.imgs_labels[i] = self.imgs_labels[i] - 1
        # TODO change folds
        self.__folds = data[0, 2]
        self.training_date_len = data[0, 3][0][0]
        self.test_data_len = data[0, 4][0][0]
        img_paths = data[0, 5][0]
        self.labels = np.array(list(np.unique(self.imgs_labels)))
        self.classes = np.array(list(map(lambda arr: arr[0], np.unique(img_paths))))
        img_names = data[0, 6][0]
        self.imgs = []
        for i, img_path in enumerate(img_paths):
            file_path = os.path.join(self.__directory, img_path[0], img_names[i][0])
            # image = ndimage.imread(file_path, mode="RGB")
            image = matplotlib.pyplot.imread(file_path)
            self.imgs.append(image)
        self.images_width = data[0, 7][0][0]
        self.input_shape = (self.images_width, self.images_width, 3)
        self.data_indexes = self.__folds[0]
