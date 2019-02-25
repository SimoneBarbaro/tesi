from execution import Execution
import models
import numpy as np


class ExperimentState:
    def __init__(self, state_info, info_name, state_number=0):
        self.__state_info = state_info
        self.__state_number = state_number
        self.__info_name = info_name
        if self.is_valid_state():
            self.__info = self.__state_info[self.__state_number]

    def get_info(self):
        return {self.__info_name: self.__info}

    def num_states(self):
        return len(self.__state_info)

    def get_state_number(self):
        return self.__state_number

    def is_valid_state(self):
        return self.__state_number < len(self.__state_info)

    def get_start(self):
        return ExperimentState(self.__state_info, self.__info_name)

    def next(self):
        return ExperimentState(self.__state_info, self.__info_name, self.__state_number + 1)


class StateDecorator(ExperimentState):
    def __init__(self, state_info, info_name, state_number=0, state: ExperimentState = None):
        super(StateDecorator, self).__init__(state_info, info_name, state_number)
        self.__inner_state = state

    def get_info(self):
        return {**super(StateDecorator, self).get_info(), **self.__inner_state.get_info()}

    def num_states(self):
        return self.__inner_state.num_states() * len(self.__state_info)

    def get_state_number(self):
        return self.__inner_state.get_state_number() + \
               self.__inner_state.num_states() * super(StateDecorator, self).get_state_number()

    def is_valid_state(self):
        return self.__inner_state.is_valid_state() and super(StateDecorator, self).is_valid_state()

    def get_start(self):
        return StateDecorator(self.__state_info, self.__info_name, state=self.__inner_state.get_start())

    def next(self):
        if self.__inner_state.next().is_valid_state():
            next_state = self.__state_number + 1
            next_inner_state = self.__inner_state.next()
        else:
            next_state = self.__state_number
            next_inner_state = self.__inner_state.get_start()
        return StateDecorator(self.__state_info, self.__info_name, next_state, next_inner_state)


class Experiment:
    def __init__(self, model_type, dataset, epochs, output_file, initial_state, metrics):
        self.model_type = model_type
        self.dataset = dataset
        self.epochs = epochs
        self.output_file = output_file
        self.state = initial_state
        self.metrics = metrics

    def resume(self):
        while self.state.is_valid_state():
            evals = []
            for f in range(0, len(self.dataset.folds)):
                data = self.dataset.get_data(f, self.model_type.get_min_input_width())
                model = models.create_model(self.model_type, data.input_shape, self.metrics)
                # TODO is this what I want?
                execution = Execution(model, data, self.state.get_info()["batch_size"], self.epochs)
                execution.run()  # TODO save checkpoints and evaluate with callbacks?
                evals.append(execution.evaluate())
            self.output_file.write(str(self.state.get_info()) + " " +
                                   str(merge_results(evals)) + "\n")  # TODO metrics?
            self.state = self.state.next()


def merge_results(results):
    return np.mean(np.array(results), 0)
