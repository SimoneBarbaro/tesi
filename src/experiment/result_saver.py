import sys
import os


class ResultSaver:

    def __init__(self, run_name, result_dir=None):
        if result_dir is None:
            self.__output_file = "stdout"
            self.__model_dir = None
            self.__confusion_dir = None
        else:
            self.__output_file = os.path.join(result_dir, run_name + ".txt")
            self.__test_output_file = os.path.join(result_dir, run_name + "_test.txt")
            self.__model_dir = os.path.join(result_dir, "models", run_name)
            self.__log_dir = os.path.join(result_dir, "logs", run_name)
            self.__confusion_dir = os.path.join(result_dir, "confusion_matrices", run_name)
            os.makedirs(self.__model_dir, exist_ok=True)
            os.makedirs(self.__confusion_dir, exist_ok=True)
            os.makedirs(self.__log_dir, exist_ok=True)

    @staticmethod
    def write_to_stdout(result):
        sys.stdout.write(str(result) + "\n")

    def write_to_output_file(self, result, test=False):
        if self.__output_file != 'stdout' or test:
            if test:
                file = open(self.__test_output_file, 'a')
            else:
                file = open(self.__output_file, 'a')
        else:
            file = sys.stdout
        file.write(str(result) + "\n")
        if file is not sys.stdout:
            file.close()

    def get_log_dir(self):
        return self.__log_dir

    def can_save_model(self):
        return self.__model_dir is not None

    def get_confusion_file(self, name):
        return os.path.join(self.__confusion_dir, name + ".png")

    def get_model_file(self, name):
        return os.path.join(self.__model_dir, name + "_model.h5")

    def get_num_states_done(self):
        if os.path.isfile(self.__output_file):
            return sum(1 for _ in open(self.__output_file, 'r'))
        else:
            return 0
