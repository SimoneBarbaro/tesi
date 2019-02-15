from google.colab import drive
import json
from models import ModelType
import experiment

drive.mount('/content/drive/')
with open('/content/drive/My Drive/Colab Notebooks/data/inception', 'w') as file:
    result = experiment.experiment_model(ModelType.INCEPTION_V3)
    result = {'inception_results': result}
    file.write(json.dumps(result))
with open('/content/drive/My Drive/Colab Notebooks/data/resnet', 'w') as file:
    result = experiment.experiment_model(ModelType.RES_NET_50)
    result = {'resnet_results': result}
    file.write(json.dumps(result))
with open('/content/drive/My Drive/Colab Notebooks/data/densenet', 'w') as file:
    result = experiment.experiment_model(ModelType.DENSE_NET_121)
    result = {'densenet_results': result}
    file.write(json.dumps(result))

