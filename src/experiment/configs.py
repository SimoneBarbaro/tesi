from src.experiment.state import ExperimentState, StateDecorator


class Config:
    def __init__(self, confing_data):
        self.models = confing_data["models"]
        self.batch_sizes = confing_data["batch_sizes"]
        self.epochs = confing_data["epochs"]
        self.preprocessing = confing_data["preprocessing"]

    def get_initial_state(self):
        return StateDecorator(self.preprocessing, "preprocessing",
                              state=ExperimentState(self.batch_sizes, "batch_size"))
