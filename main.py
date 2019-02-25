import sys
from data import Dataset
from experiment import Experiment, ExperimentState

dataset = Dataset()
num_lines = sum(1 for line in open(sys.argv[2]))
s = ExperimentState([1, 3], "batch_size")
with open(sys.argv[2], 'a') as file:
    Experiment(sys.argv[1], dataset, 1, file, num_lines - 1, ['accuracy']).resume()
