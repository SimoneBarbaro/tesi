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
        self.model_pretraining = confing_data["model"].get("pretraining", None)
        self.batch_sizes = confing_data["batch_sizes"]
        self.epochs = confing_data["epochs"]
        self.max_epochs = max(self.epochs)
        self.metrics = confing_data["metrics"]
        self.preprocessing = confing_data.get("preprocessing", [None])
        self.augmentation = confing_data.get("augmentation", [None])
        self.preprocessing_args = confing_data.get("preprocessing_args", dict())

    def has_pretraining(self):
        return isinstance(self.model_pretraining, dict)

    def fill_pretraining_data(self, data):
        if self.has_pretraining():
            data["dataset"] = self.model_pretraining["dataset"]
            data["protocol_type"] = self.model_pretraining.get("protocol_type", "full")
            data["model"] = {"name": self.model_name, "freeze": self.freeze_model,
                             "pretraining": self.model_pretraining.get("pretraining", None)
                             }
            data["batch_sizes"] = self.model_pretraining["batch_sizes"]
            data["epochs"] = self.model_pretraining["epochs"]
            data["metrics"] = self.metrics
