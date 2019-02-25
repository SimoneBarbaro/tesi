from tensorflow import keras


class ExperimentModel:
    def __init__(self, model: keras.Model, metrics, freeze: bool):
        if freeze:
            self.__freeze_model(model)
        inp = keras.Input(shape=model.input_shape)
        out = model(inp)
        out = keras.layers.Flatten()(out)
        out = keras.layers.Dense(512, activation='relu')(out)  # from paper
        self.model = keras.layers.Dense(model.output_shape, activation='softmax')(out)
        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.000001, nesterov=True),
                           loss='sparse_categorical_crossentropy', metrics=metrics, name=model.name)

    @staticmethod
    def __freeze_model(model: keras.Model):
        for layer in model.layers:
            layer.trainable = False


class PretrainedResnet50(ExperimentModel):
    def __init__(self, input_shape, metrics):
        super(PretrainedResnet50, self).__init__(keras.applications.ResNet50(weights='imagenet',
                                                                             include_top=False,
                                                                             input_shape=input_shape,
                                                                             name='resnet'), metrics, False)


class FrozenResnet50(ExperimentModel):
    def __init__(self, input_shape, metrics):
        super(FrozenResnet50, self).__init__(keras.applications.ResNet50(weights='imagenet',
                                                                         include_top=False,
                                                                         input_shape=input_shape,
                                                                         name='frozen resnet'), metrics, True)


def create_model(name, input_shape, metrics):
    if name == "resnet":
        return PretrainedResnet50(input_shape, metrics)
    elif name == "frozen resnet":
        return FrozenResnet50(input_shape, metrics)
    return None
