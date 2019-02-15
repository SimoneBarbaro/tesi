from models import ModelType
import experiment

print(experiment.experiment_model(ModelType.INCEPTION_V3))
print(experiment.experiment_model(ModelType.RES_NET_50))
print(experiment.experiment_model(ModelType.DENSE_NET_121))

