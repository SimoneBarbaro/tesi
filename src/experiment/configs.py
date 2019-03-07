from src.experiment.state import ExperimentState
from src.data.data import Dataset
from src.data.data import DataFactory


class Config:
    def __init__(self, confing_data):
        self.dataset = Dataset(confing_data["dataset"])
        self.num_folds = len(self.dataset.folds)
        self.data_factory = DataFactory(self.dataset)
        self.model_name = confing_data["model"]
        self.batch_sizes = confing_data["batch_sizes"]
        self.epochs = confing_data["epochs"]
        self.max_epochs = max(self.epochs)
        self.metrics = confing_data["metrics"]
        self.preprocessing = confing_data["preprocessing"]

    def get_initial_state(self):
        return ExperimentState(self)
