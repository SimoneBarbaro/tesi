from experiment.execution import Execution
from experiment.state import ExperimentState
from experiment.configs import Config
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from tensorflow import keras
import os
from experiment.result_saver import ResultSaver


class CheckProgressCallback(keras.callbacks.Callback):
    def __init__(self, epochs_to_check, evals, evaluate):
        super(CheckProgressCallback, self).__init__()
        self.epochs_to_check = epochs_to_check
        self.evals = evals
        self.evaluate = evaluate

    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1
        if epoch in self.epochs_to_check:
            # print("epoch " + str(epoch))
            self.evals[self.epochs_to_check.index(epoch)].append(self.evaluate())


class Experiment:

    PRETRAINING_RUN_NAME = "_pretraining"

    def __init__(self, confing: Config, saver: ResultSaver, save_testing=False):
        self.config = confing
        self.state = ExperimentState(self.config)
        self.state = self.state.init_state_number((saver.get_num_states_done() - 1) // len(self.config.epochs))
        self.result_saver = saver
        self.save_testing = save_testing

    def resume(self):
        while self.state.is_valid_state():
            run_name = str(self.state.get_state_number())
            if self.config.has_pretraining() and not os.path.exists(
                    self.result_saver.get_model_file(run_name + Experiment.PRETRAINING_RUN_NAME)):
                pre_state = self.state.get_pretraining_state()
                self.run_state(run_name + Experiment.PRETRAINING_RUN_NAME, pre_state,
                               save_output=False, fold_names=False)

            self.run_state(run_name, self.state, save_testing=self.save_testing)
            self.state = self.state.next()

    def run_state(self, run_name, state: ExperimentState, save_output=True,
                  save_log=True, save_model=True, fold_names=True, save_testing=False):
        config = state.config
        evals = [[] for _ in range(len(config.epochs))]
        test_data = []
        f = 0
        confusion = np.zeros((config.dataset.num_classes, config.dataset.num_classes), dtype=np.int32)
        for data in state.next_data():
            save_name = run_name + "_" + str(f) if fold_names else run_name
            f += 1
            if os.path.exists(self.result_saver.get_model_file(save_name)):
                model = state.load_model(self.result_saver.get_model_file(save_name))
                execution = Execution(model, data, state.batch_size, config.max_epochs)
                evals[-1].append(execution.evaluate())
            else:
                pre_model_file = None
                if config.has_pretraining():
                    pre_model_file = self.result_saver.get_model_file(run_name + Experiment.PRETRAINING_RUN_NAME)
                model = state.create_model(pre_model_file)
                execution = Execution(model, data, state.batch_size, config.max_epochs)

                callbacks = [CheckProgressCallback(config.epochs, evals, execution.evaluate)]
                if save_log and self.result_saver.get_log_dir() is not None:
                    log_dir = os.path.join(self.result_saver.get_log_dir(), save_name)
                    os.makedirs(log_dir, exist_ok=True)
                    callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir))

                execution.run(callbacks)  # returns a history
                confusion += np.array(model.confusion_matrix(data.validation_x, data.validation_y))

                if save_model and self.result_saver.can_save_model():
                    model.save(self.result_saver.get_model_file(save_name), self.config.has_pretraining())

            test_data.append(execution.test())

        dic = state.get_info().copy()
        for i, ev in enumerate(evals):
            dic["epochs"] = config.epochs[i]
            dic["data"] = "validation"
            dic["result"] = merge_results(["loss"] + config.metrics, ev)
            if save_output:
                self.result_saver.write_to_output_file(dic)

        self.save_confusion_matrix(confusion, self.result_saver.get_confusion_file(run_name))

        dic = state.get_info().copy()
        if save_testing:
            dic["epochs"] = config.epochs[-1]
            dic["data"] = "testing"
            dic["result"] = merge_results(["loss"] + config.metrics, test_data)
            self.result_saver.write_to_output_file(dic, test=True)

    def save_confusion_matrix(self, matrix, file: str):
        def insert_totals(df_cm):
            """ insert total column and line (the last ones) """
            sum_col = []
            for c in df_cm.columns:
                sum_col.append(df_cm[c].sum())
            sum_lin = []
            for item_line in df_cm.iterrows():
                sum_lin.append(item_line[1].sum())
            df_cm['total'] = sum_lin
            sum_col.append(np.sum(sum_lin))
            df_cm.loc['total'] = sum_col

        labels = self.config.dataset.classes
        cm = pd.DataFrame(matrix, index=labels, columns=labels)
        insert_totals(cm)
        ax = sn.heatmap(cm, annot=True, fmt="d")
        ax.set_title('Confusion matrix')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        plt.savefig(file)
        plt.clf()


def merge_results(metrics, results):
    return dict(zip(metrics, np.mean(np.array(results), 0)))
