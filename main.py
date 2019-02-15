from google.colab import drive
from models import ModelType
from data import Dataset
import experiment

dataset = Dataset()
drive.mount('/content/drive/')
with open('/content/drive/My Drive/Colab Notebooks/data/inception', 'w') as file:
    experiment.experiment_model(ModelType.INCEPTION_V3, dataset, file)
with open('/content/drive/My Drive/Colab Notebooks/data/resnet', 'w') as file:
    experiment.experiment_model(ModelType.RES_NET_50, dataset, file)
with open('/content/drive/My Drive/Colab Notebooks/data/densenet', 'w') as file:
    experiment.experiment_model(ModelType.DENSE_NET_121, dataset, file)

