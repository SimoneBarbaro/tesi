import sys
from models import ModelType
from data import Dataset
from experiment import Experiment

dataset = Dataset()

if sys.argv[1] == 'inception' or sys.argv[1] == 'all':
    num_lines = sum(1 for line in open(sys.argv[2]))
    with open(sys.argv[2], 'a') as file:
        Experiment(ModelType.INCEPTION_V3, dataset, file, num_lines - 1).resume()
if sys.argv[1] == 'resnet' or sys.argv[1] == 'all':
    num_lines = sum(1 for line in open(sys.argv[2]))
    with open(sys.argv[2], 'a') as file:
        Experiment(ModelType.RES_NET_50, dataset, file, num_lines - 1).resume()
if sys.argv[1] == 'densenet' or sys.argv[1] == 'all':
    num_lines = sum(1 for line in open(sys.argv[2]))
    with open(sys.argv[2], 'a') as file:
        Experiment(ModelType.DENSE_NET_121, dataset, file, num_lines - 1).resume()
