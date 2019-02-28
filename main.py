import sys
import json
from data import Dataset
from experiment import Experiment, ExperimentState

with open('confing.json') as f:
    config_data = json.load(f)

dataset = Dataset()
num_lines = sum(1 for line in open(sys.argv[2]))
s = ExperimentState(config_data["batch_sizes"], "batch_size")
with open(sys.argv[2], 'a') as file:
    Experiment(sys.argv[1], dataset, config_data["epochs"], file, s, ['accuracy']).resume()
