import sys
import json
from src.data.data import Dataset
from src.experiment.experiment import Experiment, ExperimentState

if __name__ == "__main__":
    with open('confing.json') as f:
        config_data = json.load(f)
    log_dir = None
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = "resnet"
    if len(sys.argv) > 2:
        file = open(sys.argv[2])
    else:
        file = sys.stdin
    if len(sys.argv) > 3:
        log_dir = sys.argv[3]

    dataset = Dataset("EILAT_data.mat")
    s = ExperimentState(config_data["batch_sizes"], "batch_size")
    Experiment(model, dataset, config_data["epochs"], file, s, ['accuracy'], log_dir).resume()

    if file is not sys.stdin:
        file.close()
