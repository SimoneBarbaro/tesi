from sklearn.model_selection import KFold
import scipy.io as sio


class Protocol:
    _folds = []

    @property
    def folds(self):
        return self._folds

    def save_folds(self, file_name):
        sio.savemat(file_name, {"data": self.folds})


class PreCalculatedProtocol(Protocol):
    def __init__(self, folds):
        self._folds = folds


class SklearnProtocol(Protocol):
    def __init__(self, num_folds, len_data, random_state=1):
        random_state = random_state
        kfold = KFold(num_folds, True, random_state)
        self._folds = kfold.split(range(len_data))
