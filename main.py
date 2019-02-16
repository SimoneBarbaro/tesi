import sys
from google.colab import drive
from models import ModelType
from data import Dataset
from experiment import Experiment

dataset = Dataset()
drive.mount('/content/drive/')

if sys.argv[1] == 'reset':
    with open('/content/drive/My Drive/Colab Notebooks/data/inception', 'w') as _:
        pass
    with open('/content/drive/My Drive/Colab Notebooks/data/resnet', 'w') as _:
        pass
    with open('/content/drive/My Drive/Colab Notebooks/data/densenet', 'w') as _:
        pass
else:
    if sys.argv[1] == 'inception' or sys.argv[1] == 'all':
        num_lines = sum(1 for line in open('/content/drive/My Drive/Colab Notebooks/data/inception'))
        with open('/content/drive/My Drive/Colab Notebooks/data/inception', 'a') as file:
            Experiment(ModelType.INCEPTION_V3, dataset, file, num_lines - 1).resume()
    if sys.argv[1] == 'resnet' or sys.argv[1] == 'all':
        num_lines = sum(1 for line in open('/content/drive/My Drive/Colab Notebooks/data/resnet'))
        with open('/content/drive/My Drive/Colab Notebooks/data/resnet', 'a') as file:
            Experiment(ModelType.RES_NET_50, dataset, file, num_lines - 1).resume()
    if sys.argv[1] == 'densenet' or sys.argv[1] == 'all':
        num_lines = sum(1 for line in open('/content/drive/My Drive/Colab Notebooks/data/densenet'))
        with open('/content/drive/My Drive/Colab Notebooks/data/densenet', 'a') as file:
            Experiment(ModelType.DENSE_NET_121, dataset, file, num_lines - 1).resume()
