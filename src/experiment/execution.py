from src.experiment.models import ExperimentModel
from src.data.data import Data


class Execution:
    def __init__(self, model: ExperimentModel, data: Data, batch_size, epochs):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs

    def run(self, callbacks=None):
        self.model.fit(self.data.training_x,
                       self.data.training_y,
                       self.batch_size,
                       self.epochs,
                       callbacks, )

    def evaluate(self):
        return self.model.evaluate(self.data.validation_x,
                                   self.data.validation_y,
                                   self.batch_size)
