from src.experiment.execution import Execution
from src.experiment.models import ModelFactory
from src.experiment.state import ExperimentState
from src.data.dataset import Dataset
from src.data.data import DataFactory
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
    def __init__(self, model_name, dataset: Dataset, epochs, output_file,
                 initial_state: ExperimentState, metrics, log_dir=None):
        self.model_name = model_name
        self.dataset = dataset
        self.epochs = epochs
        self.output_file = output_file
        self.state = initial_state
        self.metrics = metrics
        self.data_factory = DataFactory(self.dataset)
        self.callbacks = []
        self.callbacks.append(None)
        if log_dir is not None:
            self.callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir))

    def resume(self):
        while self.state.is_valid_state():
            evals = [[] for _ in range(len(self.epochs))]
            for f in range(0, len(self.dataset.folds)):
                # data = self.dataset.get_data(f, self.model_type.get_min_input_width())
                data = self.data_factory.build_data(validation_fold=f)
                model = ModelFactory.create_model(self.model_name, data.input_shape, data.num_classes, self.metrics)
                # TODO is this what I want?
                execution = Execution(model, data, self.state.get_info()["batch_size"], self.epochs[-1])
                self.callbacks[0] = CheckProgressCallback(self.epochs, evals, execution.evaluate)
                execution.run(self.callbacks)
            for i, ev in enumerate(evals):
                self.output_file.write(str(self.state.get_info()) + " " + "{epochs: " + str(self.epochs[i]) + "} " +
                                       str(merge_results(["loss"] + self.metrics, ev)) + "\n")
            self.state = self.state.next()


def merge_results(metrics, results):
    return dict(zip(metrics, np.mean(np.array(results), 0)))
