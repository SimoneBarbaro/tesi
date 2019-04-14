import sys
import os


class ResultSaver:

    def __init__(self, run_name, result_dir=None, log_dir=None):
        if result_dir is None:
            self.__output_file = "stdout"
            self.__model_dir = None
        else:
            self.__output_file = os.path.join(result_dir, run_name + ".txt")
            self.__model_dir = os.path.join(result_dir, "models", run_name)
            os.makedirs(self.__model_dir, exist_ok=True)
        self.__log_dir = log_dir

    @staticmethod
    def write_to_stdout(result):
        sys.stdout.write(str(result) + "\n")

    def write_to_output_file(self, result):
        if self.__output_file != 'stdout':
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

    def get_model_file(self, name):
        return os.path.join(self.__model_dir, name + "_model.h5")

    def get_num_states_done(self):
        if os.path.isfile(self.__output_file):
            return sum(1 for _ in open(self.__output_file, 'r'))
        else:
            return 0
