from sklearn.model_selection import KFold
import scipy.io as sio


class Protocol:

    @property
    def folds(self):
        raise NotImplementedError

    def save_folds(self, file_name):
        data = [[], []]
        for train_index, test_index in self.folds:
            data[0].append(train_index)
            data[1].append(test_index)
        sio.savemat(file_name, {"data": data})


class PreCalculatedProtocol(Protocol):
    def __init__(self, folds):
        self._folds = folds

    @property
    def folds(self):
        return self._folds


class SklearnProtocol(Protocol):
    def __init__(self, num_folds, len_data, random_state=1):
        random_state = random_state
        self.__kfold = KFold(num_folds, True, random_state)
        self.__len_data = len_data
        self._folds = self.__kfold.split(range(len_data))

    @property
    def folds(self):
        return self.__kfold.split(range(self.__len_data))


class FullProtocol(Protocol):
    def __init__(self, len_data, len_test_data):
        self.__len_data = len_data
        self.__len_test_data = len_test_data

    @property
    def folds(self):
        return [(range(self.__len_data), range(self.__len_data, self.__len_test_data))]
