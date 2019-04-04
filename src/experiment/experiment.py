from experiment.execution import Execution
from experiment.state import ExperimentState
from experiment.configs import Config
import numpy as np
from tensorflow import keras
import os
import json


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
        self.log_dir = log_dir
        self.callbacks = []
        self.callbacks.append(None)
        if log_dir is not None:
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir))

    def resume(self):
        # self.output_file.write("[")
        while self.state.is_valid_state():
            evals = [[] for _ in range(len(self.config.epochs))]
            f = 0
            for data in self.state.next_data():
                model = self.state.create_model()
                execution = Execution(model, data, self.state.batch_size, self.config.max_epochs)
                self.callbacks[0] = CheckProgressCallback(self.config.epochs, evals, execution.evaluate)
                if self.log_dir is not None:
                    log_dir = os.path.join(self.log_dir, str(self.state.get_state_number()) + "_" + str(f))
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                    self.callbacks[1] = keras.callbacks.TensorBoard(log_dir=log_dir)
                execution.run(self.callbacks)
                f += 1
            dic = self.state.get_info().copy()
            for i, ev in enumerate(evals):
                dic["epochs"] = self.config.epochs[i]
                dic["result"] = merge_results(["loss"] + self.config.metrics, ev)
                """
                self.output_file.write(json.dumps(dic))
                if i < len(evals) - 1 or self.state.next().is_valid_state():
                    self.output_file.write(", ")
                """
                self.output_file.write(str(dic) + "\n")
                """
                self.output_file.write(
                    str(self.state.get_info()) + " " + "{epochs: " + str(self.config.epochs[i]) + "} " +
                    str(merge_results(["loss"] + self.config.metrics, ev)) + "\n")
                """
            self.state = self.state.next()
        # self.output_file.write("]")


def merge_results(metrics, results):
    return dict(zip(metrics, np.mean(np.array(results), 0)))
