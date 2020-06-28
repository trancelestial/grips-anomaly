from keras.layers import (
    Dense,
    LeakyReLU,
    BatchNormalization,
    Add,
)
import keras.backend as K


def ff_relu(units, alpha=0., bn=None, **kwargs):
    def f(x):
        x = Dense(units, **kwargs)(x)
        x = LeakyReLU(alpha=alpha)(x)
        if bn is not None:
            if not isinstance(bn, bool):
                x = BatchNormalization(**bn)(x)
            elif bn:
                x = BatchNormalization()(x)
        return x
    return f


def bn_relu_ff(units, alpha=0., bn=None, **kwargs):
    def f(x):
        if bn is not None:
            if not isinstance(bn, bool):
                x = BatchNormalization(**bn)(x)
            elif bn:
                x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = Dense(units, **kwargs)(x)
        return x
    return f


def ff_block(units, size=2, bottleneck_factor=0, **kwargs):
    def f(x):
        inputs = x
        for _ in range(size):
            x = bn_relu_ff(units, **kwargs)(x)
        if bottleneck_factor:
            x = bn_relu_ff(bottleneck_factor * units, **kwargs)(x)
        x = shortcut()([inputs, x])
        return x
    return f


def shortcut(**kwargs):
    def f(x):
        inputs, outputs = x
        _, input_units = K.int_shape(inputs)
        _, output_units = K.int_shape(outputs)
        if input_units != output_units:
            inputs = Dense(output_units, **kwargs)(inputs)
        x = Add()([inputs, outputs])
        return x
    return f
