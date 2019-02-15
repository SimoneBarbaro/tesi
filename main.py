from google.colab import drive
from models import ModelType
import experiment

drive.mount('/content/drive/')
with open('/content/drive/My Drive/Colab Notebooks/data/inception', 'w') as file:
    file.write(str(experiment.experiment_model(ModelType.INCEPTION_V3)))
with open('/content/drive/My Drive/Colab Notebooks/data/resnet', 'w') as file:
    file.write(str(experiment.experiment_model(ModelType.RES_NET_50)))
with open('/content/drive/My Drive/Colab Notebooks/data/densenet', 'w') as file:
    file.write(str(experiment.experiment_model(ModelType.DENSE_NET_121)))

