from data import Dataset
from execution import Execution
import models
import numpy as np


def experiment_model(model_type):
    batch_sizes = [32, 64, 128]
    num_epochs_list = [100, 300, 500, 700, 1000, 1300]
    # TODO change after test
    batch_sizes = [32]
    num_epochs_list = [10]
    dataset = Dataset()
    result = {}
    for batch_size in batch_sizes:
        for num_epochs in num_epochs_list:
            evals = []
            for f in range(0, len(dataset.folds)):
                data = dataset.get_data(f, 75)  # min for inception
                model = models.get_model(data, model_type)
                execution = Execution(model, data, batch_size, num_epochs)
                execution.run()
                evals.append(execution.evaluate())
                break  # TODO remove after test
            result[batch_size, num_epochs] = merge_results(evals)
    return result


def merge_results(results):
    return np.mean(np.array(results), 0)
