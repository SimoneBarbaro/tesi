from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ExperimentModel:
    def __init__(self, model: keras.Model, metrics):
        self.model = model
        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.000001, nesterov=True),
                           loss='sparse_categorical_crossentropy', metrics=metrics)

    def fit(self, x, y, batch_size, epochs, callbacks, val_x, val_y):
        self.model.fit(x, y,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=callbacks,
                       validation_data=(val_x, val_y),
                       verbose=0)

    def fit_generator(self, x, y, batch_size, epochs, callbacks, val_x, val_y, generator: ImageDataGenerator):
        self.model.fit_generator(generator.flow(x, y, batch_size=batch_size),
                                 steps_per_epoch=len(x) / batch_size,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 validation_data=generator.flow(val_x, val_y, batch_size=batch_size),
                                 validation_steps=len(val_x) / batch_size)

    def evaluate(self, x, y, batch_size):
        return self.model.evaluate(x, y, batch_size=batch_size)


class TestModel(ExperimentModel):
    def __init__(self, input_shape, num_classes, metrics):
        super(TestModel, self).__init__(keras.Sequential([keras.layers.Flatten(input_shape=input_shape),
                                                          keras.layers.Dense(512, activation='relu'),
                                                          keras.layers.Dense(num_classes, activation='softmax')]),
                                        metrics)


class CompletedModel(ExperimentModel):
    def __init__(self, model: keras.Model, input_shape, num_classes, metrics, freeze: bool):
        super(CompletedModel, self).__init__(CompletedModel.complete_model(model, input_shape, num_classes, freeze),
                                             metrics)

    @staticmethod
    def freeze_model(model: keras.Model):
        for layer in model.layers:
            layer.trainable = False

    @staticmethod
    def complete_model(model: keras.Model, input_shape, num_classes, freeze: bool):
        if freeze:
            CompletedModel.freeze_model(model)

        inp = keras.Input(input_shape)
        out = model(inp)
        out = keras.layers.Flatten()(out)
        out = keras.layers.Dense(512, activation='relu')(out)
        out = keras.layers.Dense(num_classes, activation='softmax')(out)
        return keras.Model(inputs=inp, outputs=out, name=model.name + ('_frozen' if freeze else ''))


class Resnet50(CompletedModel):
    def __init__(self, input_shape, num_classes, metrics, freeze=False, pretraining="imagenet"):
        super(Resnet50, self).__init__(keras.applications.ResNet50(weights=pretraining,
                                                                   include_top=False,
                                                                   input_shape=input_shape),
                                       input_shape, num_classes, metrics, freeze)


class Densenet121(CompletedModel):
    def __init__(self, input_shape, num_classes, metrics, freeze=False, pretraining="imagenet"):
        super(Densenet121, self).__init__(keras.applications.DenseNet121(weights=pretraining,
                                                                         include_top=False,
                                                                         input_shape=input_shape),
                                          input_shape, num_classes, metrics, freeze)


class ModelFactory:

    @staticmethod
    def create_model(name, input_shape, num_classes, metrics, freeze=True, pretraining="imagenet"):
        if name == "resnet":
            return Resnet50(input_shape, num_classes, metrics, freeze, pretraining)
        elif name == "densenet":
            return Densenet121(input_shape, num_classes, metrics, freeze, pretraining)
        elif name == "test":
            return TestModel(input_shape, num_classes, metrics)
        return None
