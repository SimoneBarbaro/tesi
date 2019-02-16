from execution import Execution
import models
import numpy as np


class ExperimentState:
    batch_sizes = [32, 64, 128]
    num_epochs_list = [100, 300, 500, 700, 1000, 1300]

    def __init__(self, state_numer=0):
        self.__batch_size_index = self.batch_sizes[int(state_numer / len(self.batch_sizes))]
        self.__num_epochs_index = self.num_epochs_list[state_numer % len(self.num_epochs_list)]
        self.batch_size = self.batch_sizes[int(state_numer / len(self.batch_sizes))]
        self.num_epochs = self.num_epochs_list[state_numer % len(self.num_epochs_list)]

    def is_valid_state(self):
        return self.__batch_size_index < len(self.batch_sizes) and self.__num_epochs_index < len(self.num_epochs_list)


class Experiment:
    def __init__(self, model_type, dataset, output_file, last_state_number=-1):
        self.model_type = model_type
        self.dataset = dataset
        self.output_file = output_file
        self.state_number = last_state_number + 1

    def resume(self):
        state = ExperimentState(self.state_number)
        while state.is_valid_state():
            evals = []
            for f in range(0, len(self.dataset.folds)):
                data = self.dataset.get_data(f, self.model_type.get_min_input_width())
                model = models.get_model(data, self.model_type)
                execution = Execution(model, data, state.batch_size, state.num_epochs)
                execution.run()
                evals.append(execution.evaluate())
            self.output_file.write(str(state.batch_size) + " " +
                                   str(state.num_epochs) + " " +
                                   str(merge_results(evals)) + "\n")
            self.state_number += 1


def merge_results(results):
    return np.mean(np.array(results), 0)
