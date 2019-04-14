from data.data import Dataset
from data.data_factory import DataFactory
from data.protocol import SklearnProtocol, FullProtocol

DEFAULT_NUM_FOLDS = 5


class Config:
    def __init__(self, confing_data):
        self.dataset = Dataset(confing_data["dataset"])
        self.protocol_type = confing_data.get("protocol_type", "kfold")
        if self.protocol_type == "kfold":
            self.num_folds = confing_data.get("num_folds", DEFAULT_NUM_FOLDS)
            self.protocol = SklearnProtocol(self.num_folds, self.dataset.training_date_len)
        elif self.protocol_type == "full":
            self.protocol = FullProtocol(self.dataset.training_date_len, self.dataset.test_data_len)
        else:
            raise NotImplementedError("protocol type not implemented or not valid")
        self.data_factory = DataFactory(self.dataset)
        self.model_name = confing_data["model"]["name"]
        self.freeze_model = confing_data["model"]["freeze"]
        self.model_pretraining = confing_data["model"]["pretraining"]
        self.batch_sizes = confing_data["batch_sizes"]
        self.epochs = confing_data["epochs"]
        self.max_epochs = max(self.epochs)
        self.metrics = confing_data["metrics"]
        self.preprocessing = confing_data.get("preprocessing", [None])
        self.augmentation = confing_data.get("augmentation", [None])
