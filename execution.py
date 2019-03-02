class Execution:
    def __init__(self, model, data, batch_size, epochs):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs

    def run(self, callbacks=None):
        self.model.model.fit(self.data.training_x,
                             self.data.training_y,
                             batch_size=self.batch_size,
                             epochs=self.epochs,
                             validation_split=0.0,
                             callbacks=callbacks,
                             verbose=0)

    def evaluate(self):
        return self.model.model.evaluate(self.data.validation_x,
                                         self.data.validation_y,
                                         batch_size=self.batch_size)
