from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ExperimentModel:
    def __init__(self, model: keras.Model, metrics):
        self.model = model
        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.000001, nesterov=True),
                           loss='sparse_categorical_crossentropy', metrics=metrics)
        # self.model.summary()

    def fit(self, x, y, batch_size, epochs, callbacks, val_x, val_y):
        return self.model.fit(x, y,
                              batch_size=batch_size,
                              epochs=epochs,
                              callbacks=callbacks,
                              validation_data=(val_x, val_y),
                              verbose=0)

    def fit_generator(self, x, y, batch_size, epochs, callbacks, val_x, val_y, generator: ImageDataGenerator):
        return self.model.fit_generator(generator.flow(x, y, batch_size=batch_size),
                                        steps_per_epoch=len(x) / batch_size,
                                        epochs=epochs,
                                        callbacks=callbacks,
                                        validation_data=generator.flow(val_x, val_y, batch_size=batch_size),
                                        validation_steps=len(val_x) / batch_size)

    def save(self, file: str, only_inner=False):
        if only_inner:
            raise NotImplementedError
        # self.model.save(file)
        self.model.save_weights(file)

    def load(self, file: str):
        self.model.load_weights(file)

    def evaluate(self, x, y, batch_size):
        return self.model.evaluate(x, y, batch_size=batch_size)


class TestModel(ExperimentModel):
    def __init__(self, input_shape, num_classes, metrics):
        super(TestModel, self).__init__(keras.Sequential([keras.layers.Flatten(input_shape=input_shape),
                                                          keras.layers.Dense(64, activation='relu'),
                                                          keras.layers.Dense(num_classes, activation='softmax')
                                                          ]),
                                        metrics)


class CompletedModel(ExperimentModel):
    def __init__(self, model: keras.Model, input_shape, num_classes, metrics, freeze: bool):
        self.__inner_model = model
        super(CompletedModel, self).__init__(CompletedModel.complete_model(model, input_shape,
                                                                           num_classes, freeze),
                                             metrics)

    def save(self, file: str, only_inner=False):
        if only_inner:
            self.__inner_model.save(file)
        else:
            super(CompletedModel, self).save(file)

    @staticmethod
    def freeze_model(model: keras.Model):
        for layer in model.layers:
            layer.trainable = False

    @staticmethod
    def complete_model(model: keras.Model, input_shape, num_classes, freeze: bool):
        if freeze:
            CompletedModel.freeze_model(model)

        # inp = keras.Input((None, None, 3))
        # model.summary()
        inp = keras.Input(input_shape)
        out = model(inp)
        # out = keras.layers.Flatten()(out)
        out = keras.layers.GlobalAveragePooling2D()(out)
        out = keras.layers.Dense(512, activation='relu')(out)
        out = keras.layers.Dense(num_classes, activation='softmax')(out)
        return keras.Model(inputs=inp, outputs=out, name=model.name + ('_frozen' if freeze else ''))


class ConvolutionalTestModel(CompletedModel):
    def __init__(self, input_shape, num_classes, metrics):
        super(ConvolutionalTestModel, self).__init__(
            keras.Sequential([keras.layers.Conv2D(1, 32, input_shape=(None, None, 3)),
                              keras.layers.MaxPooling2D(pool_size=(20, 20)),
                              ]),
            input_shape, num_classes, metrics, False)


class Resnet50(CompletedModel):
    def __init__(self, input_shape, num_classes, metrics, freeze=False, pretraining="imagenet"):
        super(Resnet50, self).__init__(keras.applications.ResNet50(weights=pretraining,
                                                                   include_top=False,
                                                                   input_shape=(None, None, 3)),
                                       input_shape, num_classes, metrics, freeze)


class Densenet121(CompletedModel):
    def __init__(self, input_shape, num_classes, metrics, freeze=False, pretraining="imagenet"):
        super(Densenet121, self).__init__(keras.applications.DenseNet121(weights=pretraining,
                                                                         include_top=False,
                                                                         input_shape=(None, None, 3)),
                                          input_shape, num_classes, metrics, freeze)


class SavedModel(CompletedModel):
    def __init__(self, input_shape, num_classes, metrics, pretraining, freeze=False):
        model = keras.models.load_model(pretraining)
        # Removing final layers
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()
        super(SavedModel, self).__init__(model, input_shape, num_classes, metrics, freeze)


class ModelFactory:
    """
    @staticmethod
    def load_model(file: str, metrics):

        # return ExperimentModel(keras.models.load_model(file), metrics)
    """
    @staticmethod
    def create_model(name, input_shape, num_classes, metrics, freeze=True, pretraining="imagenet"):
        if isinstance(pretraining, str) and pretraining.endswith(".h5"):
            return SavedModel(input_shape, num_classes, metrics, pretraining=pretraining, freeze=freeze)
        elif name == "resnet":
            return Resnet50(input_shape, num_classes, metrics, pretraining=pretraining, freeze=freeze)
        elif name == "densenet":
            return Densenet121(input_shape, num_classes, metrics, pretraining=pretraining, freeze=freeze)
        elif name == "test":
            return TestModel(input_shape, num_classes, metrics)
        elif name == "convolutional_test":
            return ConvolutionalTestModel(input_shape, num_classes, metrics)
        return None
