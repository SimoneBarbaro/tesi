from experiment.configs import Config
from experiment.model import ModelFactory


class BlankState:
    def __init__(self, state_info, info_name, state_number=0):
        self._state_info = state_info
        self._state_number = state_number
        self._info_name = info_name
        self.__info = None
        if self._state_number < len(self._state_info):
            self.__info = self._state_info[self._state_number]

    def init_state_number(self, number):
        self._state_number = number
        return self.next()

    def get_info(self):
        return {self._info_name: self.__info}

    def num_states(self):
        return len(self._state_info)

    def get_state_number(self):
        return self._state_number

    def is_valid_state(self):
        return self._state_number < len(self._state_info)

    def get_start(self):
        return BlankState(self._state_info, self._info_name)

    def next(self):
        return BlankState(self._state_info, self._info_name, self._state_number + 1)


class StateDecorator(BlankState):
    def __init__(self, state_info, info_name, state_number=0, state: BlankState = None):
        super(StateDecorator, self).__init__(state_info, info_name, state_number)
        self.__inner_state = state

    def init_state_number(self, number):
        self._state_number = number // self.__inner_state.num_states()
        self.__inner_state = self.__inner_state.init_state_number(number % self.__inner_state.num_states())
        return self.next()

    def get_info(self):
        return {**super(StateDecorator, self).get_info(), **self.__inner_state.get_info()}

    def num_states(self):
        return self.__inner_state.num_states() * len(self._state_info)

    def get_state_number(self):
        return self.__inner_state.get_state_number() + \
               self.__inner_state.num_states() * super(StateDecorator, self).get_state_number()

    def is_valid_state(self):
        return self.__inner_state.is_valid_state() and super(StateDecorator, self).is_valid_state()

    def get_start(self):
        return StateDecorator(self._state_info, self._info_name, state=self.__inner_state.get_start())

    def _create_next(self, next_state, next_inner_state):
        return StateDecorator(self._state_info, self._info_name, next_state, next_inner_state)

    def next(self):
        if self.__inner_state.next().is_valid_state():
            next_state = self._state_number
            next_inner_state = self.__inner_state.next()
        else:
            next_state = self._state_number + 1
            next_inner_state = self.__inner_state.get_start()
        return self._create_next(next_state, next_inner_state)


class ExperimentState(StateDecorator):
    def __init__(self, config: Config, state_number=0, state: BlankState = None):
        super(ExperimentState, self).__init__(config.preprocessing, "preprocessing",
                                              state_number=state_number,
                                              state=StateDecorator(config.augmentation, "augmentation",
                                                                   state=BlankState(config.batch_sizes, "batch_size"))
                                              if state is None
                                              else state)
        # self._init_state_number(state_number)
        self.config = config
        self.preprocessing = self.get_info()["preprocessing"]
        self.augmentation = self.get_info()["augmentation"]
        self.batch_size = self.get_info()["batch_size"]
        self._data = None

    def next_data(self):
        for train_index, test_index in self.config.protocol.folds:

            self._data = self.config.data_factory.build_data(train_index, test_index,
                                                             preprocessing=self.preprocessing,
                                                             augmentation=self.augmentation)
            yield self._data

    def create_model(self):
        return ModelFactory.create_model(self.config.model_name, self._data.input_shape,
                                         self._data.num_classes, self.config.metrics,
                                         self.config.freeze_model, self.config.model_pretraining)

    def _create_next(self, next_state, next_inner_state):
        return ExperimentState(self.config, next_state, next_inner_state)


"""
class CVState(ExperimentState):
    def __init__(self, config: Config, state_number=0, state: BlankState = None, random_state=1):
        super(CVState, self).__init__(config, state_number, state)
        self.random_state = random_state
        self.kfold = KFold(config.num_folds, True, self.random_state)

    def next_data(self):
        for train_index, test_index, in self.kfold.split(range(self.config.dataset.dim1)):
            self._data = self.config.data_factory.build_data(train_index=train_index,
                                                             test_index=test_index,
                                                             preprocessing=self.preprocessing,
                                                             augmentation=self.augmentation)
            yield self._data

    def _create_next(self, next_state, next_inner_state):
        return CVState(self.config, next_state, next_inner_state, self.random_state)
"""