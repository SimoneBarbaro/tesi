import os
import sys
import json
from experiment.configs import Config
from experiment.experiment import Experiment
from experiment.result_saver import ResultSaver

if __name__ == "__main__":
    log_dir = None
    # start_state = 0
    if len(sys.argv) > 1:
        run_name = sys.argv[1]
    else:
        run_name = "test"
    if len(sys.argv) > 2 and os.path.isfile(sys.argv[2]):
        res_dir = sys.argv[2]
    else:
        res_dir = None
    if len(sys.argv) > 3:
        log_dir = sys.argv[3]

    config_file = run_name + '_config.json'
    with open(os.path.join(os.path.dirname(__file__), os.pardir, 'config', config_file)) as f:
        config_data = json.load(f)
    config = Config(config_data)

    # config.protocol.save_folds("folds.mat")
    Experiment(config, ResultSaver(run_name, log_dir, res_dir)).resume()

