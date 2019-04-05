import os
import sys
import json
from experiment.configs import Config
from experiment.experiment import Experiment

if __name__ == "__main__":
    log_dir = None
    start_state = 0
    if len(sys.argv) > 1:
        start_state = sum(1 for line in open(sys.argv[1], 'r'))
        file = open(sys.argv[1], 'w')
    else:
        file = sys.stdout
    if len(sys.argv) > 2:
        config_file = sys.argv[2] + "_config.json"
    else:
        config_file = 'test_config.json'
    if len(sys.argv) > 3:
        log_dir = sys.argv[3]

    with open(os.path.join(os.path.dirname(__file__), os.pardir, 'config', config_file)) as f:
        config_data = json.load(f)
    config = Config(config_data)

    # config.protocol.save_folds("folds.mat")

    Experiment(config, file, start_state, log_dir).resume()

    if file is not sys.stdin:
        file.close()
