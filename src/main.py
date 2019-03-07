import sys
import json
from src.experiment.configs import Config
from src.experiment.experiment import Experiment

if __name__ == "__main__":
    with open('confing.json') as f:
        config_data = json.load(f)
    log_dir = None
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = "test"
    if len(sys.argv) > 2:
        file = open(sys.argv[2])
    else:
        file = sys.stdin
    if len(sys.argv) > 3:
        log_dir = sys.argv[3]

    config = Config(config_data)
    Experiment(config, file, log_dir).resume()

    if file is not sys.stdin:
        file.close()
