import sys
import os


class ResultSaver:

    def __init__(self, run_name, log_dir=None, result_dir=None):
        if result_dir is None:
            self.__output_file = "stdout"
            self.__model_file = None
        else:
            self.__output_file = os.path.join(result_dir, run_name + ".txt")
            self.__model_file = os.path.join(result_dir, run_name + "_model.h5")
        self.__log_dir = log_dir

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

    def get_model_file(self):
        return self.__model_file

    def get_num_states_done(self):
        if os.path.isfile(self.__output_file):
            return sum(1 for _ in open(self.__output_file, 'r'))
        else:
            return 0
