from tensorflow import keras


class ExperimentModel:
    def __init__(self, model: keras.Model, input_shape, num_classes, metrics, freeze: bool):
        if freeze:
            self.__freeze_model(model)

        inp = keras.Input(input_shape)
        out = model(inp)
        out = keras.layers.Flatten()(out)
        out = keras.layers.Dense(512, activation='relu')(out)  # from paper
        out = keras.layers.Dense(num_classes, activation='softmax')(out)
        # self.model = keras.Model(inputs=inp, outputs=out, name=model.name + ('_frozen' if freeze else ''))
        self.model = keras.Model(inputs=inp, outputs=out, name=model.name)
        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.000001, nesterov=True),
                           loss='sparse_categorical_crossentropy', metrics=metrics)

    @staticmethod
    def __freeze_model(model: keras.Model):
        for layer in model.layers:
            layer.trainable = False

    def fit(self, x, y, batch_size, epochs, callbacks):
        self.model.fit(x, y,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=callbacks,
                       verbose=0)

    def evaluate(self, x, y, batch_size):
        self.model.evaluate(x, y, batch_size=batch_size)


class Resnet50(ExperimentModel):
    def __init__(self, input_shape, num_classes, metrics):
        super(Resnet50, self).__init__(keras.applications.ResNet50(weights='imagenet',
                                                                             include_top=False,
                                                                             input_shape=input_shape),
                                                 input_shape, num_classes, metrics, False)


class ModelFactory:

    @staticmethod
    def create_model(name, input_shape, num_classes, metrics):
        if name == "resnet":
            return Resnet50(input_shape, num_classes, metrics)
        return None
