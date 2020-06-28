import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Model
from keras.layers import (
    Input,
    Embedding,
    Dense,
    ReLU,
    Flatten,
)
from sklearn.utils import class_weight

from scripts.util import load_credit_card_data


def to_dataset(creditcard_data):
    x = (creditcard_data['Time'].astype('int32') // 3600) % 24
    y = creditcard_data['Class']
    return x, y


if __name__ == '__main__':
    seed = 12345

    np.random.seed(seed)
    tf.set_random_seed(seed)

    train, test = load_credit_card_data('../data', mode='train_test')

    train_x, train_y = to_dataset(train)
    test_x, test_y = to_dataset(test)

    cls_weights = class_weight.compute_class_weight('balanced', [0, 1], train_y)
    cls_weights = {i: weight for i, weight in enumerate(cls_weights)}

    num_features = len(np.unique(train_x))
    embedding_size = 2  # min(num_features // 2, 50)
    hidden = (100, 50)

    in_ = x = Input(shape=(1,))
    x = Embedding(num_features, embedding_size)(x)
    x = z = Flatten()(x)
    for units in hidden:
        x = Dense(units)(x)
        x = ReLU()(x)
    out_ = Dense(1, activation='sigmoid')(x)
    model = Model(in_, out_)

    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=64, epochs=20, validation_split=0.2)

    encoder = Model(in_, z)

    embedding = encoder.predict(np.unique(train_x))

    np.save('../output/time_embedding.npy', embedding)

    if embedding_size == 2:
        plt.scatter(embedding[:, 0], embedding[:, 1])
        plt.savefig('embedding.png')
