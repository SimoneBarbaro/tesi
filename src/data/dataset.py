import os
import scipy.io
from scipy import ndimage


class Dataset:

    def __init__(self, mat_file):
        mat = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, mat_file))
        data = mat["data"]
        self.__directory = os.path.join(data[0, 0][0][0][0], data[0, 0][0][1][0])
        self.labels = data[0, 1][0]
        self.num_classes = max(self.labels)
        for i in range(0, len(self.labels)):
            self.labels[i] = self.labels[i] - 1
        # TODO change folds
        self.__folds = data[0, 2]
        self.training_date_len = data[0, 3][0][0]
        self.test_data_len = data[0, 4][0][0]
        img_paths = data[0, 5][0]
        img_names = data[0, 6][0]
        self.imgs = []
        for i, img_path in enumerate(img_paths):
            file_path = os.path.join(self.__directory, img_path[0], img_names[i][0])
            image = ndimage.imread(file_path, mode="RGB")
            self.imgs.append(image)
        self.images_width = data[0, 7][0][0]
        self.input_shape = (self.images_width, self.images_width, 3)
        self.data_indexes = self.__folds[0]
