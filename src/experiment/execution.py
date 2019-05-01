from experiment.model import ExperimentModel
from data.data import Data
from data.preprocessing import AugmentedData


class Execution:
    def __init__(self, model: ExperimentModel, data: Data, batch_size, epochs):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs

    def run(self, callbacks=None):
        if self.data is AugmentedData:
            return self.model.fit_generator(self.data.training_x,
                                            self.data.training_y,
                                            self.batch_size,
                                            self.epochs,
                                            callbacks,
                                            self.data.validation_x,
                                            self.data.validation_y,
                                            self.data.get_generator())
        else:
            return self.model.fit(self.data.training_x,
                                  self.data.training_y,
                                  self.batch_size,
                                  self.epochs,
                                  callbacks,
                                  self.data.validation_x,
                                  self.data.validation_y)

    def evaluate(self):
        return self.model.evaluate(self.data.validation_x,
                                   self.data.validation_y,
                                   self.batch_size)

    def test(self):
        return self.model.evaluate(self.data.testing_x,
                                   self.data.testing_y,
                                   self.batch_size)
