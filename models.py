from tensorflow import keras

from enum import Enum


def inception_v3_sub_model(input_shape):
    return keras.applications.InceptionV3(weights='imagenet',
                                          include_top=False,
                                          input_shape=input_shape)


def resnet50_sub_model(input_shape):
    return keras.applications.ResNet50(weights='imagenet',
                                       include_top=False,
                                       input_shape=input_shape)


def densenet121_sub_model(input_shape):
    return keras.applications.DenseNet121(weights='imagenet',
                                          include_top=False,
                                          input_shape=input_shape)


class ModelType(Enum):
    INCEPTION_V3 = inception_v3_sub_model
    RES_NET_50 = resnet50_sub_model
    DENSE_NET_121 = densenet121_sub_model

# miss one dense net and resnet


def freeze_model(model):
    for layer in model.layers:
        layer.trainable = False


# num_classes = 8 for EILAT
def complete_model(model, input_shape, num_classes):
    freeze_model(model)
    inp = keras.Input(shape=input_shape)
    out = model(inp)
    out = keras.layers.Flatten()(out)
    out = keras.layers.Dense(512, activation='relu')(out)  # from paper
    out = keras.layers.Dense(num_classes, activation='softmax')(out)
    return keras.Model(inputs=inp, outputs=out)


def compile_model(model, metrics):
    model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=0.000001, nesterov=True),  # from paper
                  loss='sparse_categorical_crossentropy',
                  metrics=metrics)


def get_model(data, model_type):
    if model_type is not None:
        model = complete_model(model_type(data.input_shape),
                               data.input_shape, data.num_classes)
        compile_model(model, ['accuracy'])
        return model
    else:
        return None
