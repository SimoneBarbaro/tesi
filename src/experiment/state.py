from experiment.configs import Config
from experiment.model import ModelFactory


class BlankState:
    def __init__(self, state_info, info_name, state_number=0):
        self.__state_info = state_info
        self.__state_number = state_number
        self.__info_name = info_name
        if self.__state_number < len(self.__state_info):
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
        return BlankState(self.__state_info, self.__info_name)

    def next(self):
        return BlankState(self.__state_info, self.__info_name, self.__state_number + 1)


class StateDecorator(BlankState):
    def __init__(self, state_info, info_name, state_number=0, state: BlankState = None):
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


class ExperimentState(StateDecorator):
    def __init__(self, config: Config):
        super(ExperimentState, self).__init__(config.preprocessing, "preprocessing",
                                              state=BlankState(config.batch_sizes, "batch_size"))
        self.config = config
        self.preprocessing = self.get_info()["preprocessing"]
        self.batch_size = self.get_info()["batch_size"]
        self.current_fold = -1
        self.data = None

    def next_data(self):
        if self.current_fold + 1 < self.config.num_folds:
            self.current_fold = self.current_fold + 1
            self.data = self.config.data_factory.build_data(self.current_fold, self.preprocessing)
            return True
        return False

    def create_model(self):
        if self.data is not None:
            return ModelFactory.create_model(self.config.model_name, self.data.input_shape,
                                             self.data.num_classes, self.config.metrics)
