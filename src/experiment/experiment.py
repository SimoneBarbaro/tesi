from experiment.execution import Execution
from experiment.state import ExperimentState
from experiment.configs import Config
import numpy as np
from tensorflow import keras


class CheckProgressCallback(keras.callbacks.Callback):
    def __init__(self, epochs_to_check, evals, evaluate):
        super(CheckProgressCallback, self).__init__()
        self.epochs_to_check = epochs_to_check
        self.evals = evals
        self.evaluate = evaluate

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1
        if epoch in self.epochs_to_check:
            print("epoch " + str(epoch))
            self.evals[self.epochs_to_check.index(epoch)].append(self.evaluate())


class Experiment:

    def __init__(self, confing: Config, output_file, log_dir=None):
        self.config = confing
        self.state = ExperimentState(self.config)
        self.output_file = output_file
        self.callbacks = []
        self.callbacks.append(None)
        if log_dir is not None:
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir))

    def resume(self):
        while self.state.is_valid_state():
            evals = [[] for _ in range(len(self.config.epochs))]
            while self.state.next_data():
                # data = self.dataset.get_data(f, self.model_type.get_min_input_width())
                data = self.state.data
                model = self.state.create_model()
                # TODO is this what I want?
                execution = Execution(model, data, self.state.batch_size, self.config.max_epochs)
                self.callbacks[0] = CheckProgressCallback(self.config.epochs, evals, execution.evaluate)
                execution.run(self.callbacks)
            for i, ev in enumerate(evals):
                self.output_file.write(
                    str(self.state.get_info()) + " " + "{epochs: " + str(self.config.epochs[i]) + "} " +
                    str(merge_results(["loss"] + self.config.metrics, ev)) + "\n")
            self.state = self.state.next()


def merge_results(metrics, results):
    return dict(zip(metrics, np.mean(np.array(results), 0)))
