from scripts.util import load_credit_card_data
from keras.layers import (
    Dense,
    LeakyReLU,
    Dropout,
    Input
)
from keras.models import Model
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint
import keras.backend as K
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)
from keras.models import load_model
import pandas as pd


def mse(y_true, y_pred, axis=None):
    return np.square(y_pred - y_true).mean(axis=axis)


def to_dataset(creditcard_data):
    x = creditcard_data.drop(columns=['Time', 'Class', 'Amount'])
    # creditcard_data['Amount'] = np.log(creditcard_data['Amount'] + 1)
    y = creditcard_data['Class']
    return x, y


class PerformanceCallback(Callback):
    def __init__(self, train_data, validation_data):
        super(PerformanceCallback, self).__init__()
        self._train_data = train_data
        self._validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        train_x, train_y = self._train_data
        val_x, val_y = self._validation_data
        train_x_reconstructed = self.model.predict(train_x)
        val_x_reconstructed = self.model.predict(val_x)
        train_loss_per_instance = mse(train_x, train_x_reconstructed, axis=1).reshape((-1, 1))
        scaler = MinMaxScaler().fit(train_loss_per_instance)
        val_loss_per_instance = mse(val_x, val_x_reconstructed, axis=1)
        normalized_val_loss_per_instance = scaler.transform(val_loss_per_instance.reshape((-1, 1))).clip(0, 1)
        val_auc_pr = average_precision_score(val_y.values.reshape((-1, 1)), normalized_val_loss_per_instance)
        val_auc_roc = roc_auc_score(val_y.values.reshape((-1, 1)), normalized_val_loss_per_instance)
        print(f'[Epoch {epoch + 1}] AUC-PR: {val_auc_pr:0.4f} AUC-ROC: {val_auc_roc:0.4f}')


class NoisyDataGenerator(Sequence):
    def __init__(self, x, loc=0., scale=1., batch_size=1, shuffle=True):
        self.x = x
        self.loc = loc
        self.scale = scale
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_examples, self.num_features = x.shape
        self._index = np.arange(self.num_examples)

    def __getitem__(self, index):
        indices = self._index[index * self.batch_size:(index + 1) * self.batch_size]
        noise = np.random.normal(loc=self.loc, scale=self.scale, size=(self.batch_size, self.num_features))
        x_batch = self.x[indices]
        x_batch_sigma = x_batch + noise
        return x_batch, x_batch_sigma

    def __len__(self):
        return self.num_examples // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._index)


def ff_relu(units, alpha=0.):
    def f(x):
        x = Dense(units)(x)
        x = LeakyReLU(alpha=alpha)(x)
        return x
    return f


if __name__ == '__main__':
    seed = 12345
    train, test = load_credit_card_data('../data', mode='train_test')

    train_x, train_y = to_dataset(train)
    test_x, test_y = to_dataset(test)

    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    _, num_features = train_x.shape
    hidden = (128, 64)
    latent_dim = 16
    alpha = 0.

    in_ = x = Input((num_features,))
    for units in hidden:
        x = ff_relu(units, alpha)(x)
    x = z = Dense(latent_dim, name='embedding')(x)
    for units in hidden[::-1]:
        x = ff_relu(units, alpha)(x)
    out_ = Dense(num_features, activation='linear')(x)
    model = Model(in_, out_)

    model.summary()

    model.compile(optimizer='adam', loss='mse')

    # load_model('autoencoder.h5')

    # model.fit_generator(NoisyDataGenerator(train_x, scale=0.1, batch_size=64), epochs=100, verbose=2,
    #                     callbacks=[TensorBoard(), ModelCheckpoint('best_model.h5', save_best_only=True, verbose=1,
    #                                                               monitor='loss')])

    train_x_norm = train_x[train_y == 0]

    model.fit(train_x_norm, train_x_norm, batch_size=64, epochs=40, verbose=2, validation_split=0.2,
              callbacks=[ModelCheckpoint('best_model.h5', save_best_only=True, verbose=1, monitor='val_loss')])

    model.save('autoencoder.h5')
    model = load_model('best_model.h5')

    encoder = Model(in_, z)

    train_xhat = encoder.predict(train_x)
    test_xhat = encoder.predict(test_x)

    pd.DataFrame(data=train_xhat,
                 index=train.index,
                 columns=['E%d' % (i + 1) for i in range(latent_dim)]).to_csv('train_embedding_%dd.csv' % latent_dim)

    pd.DataFrame(data=test_xhat,
                 index=test.index,
                 columns=['E%d' % (i + 1) for i in range(latent_dim)]).to_csv('test_embedding_%dd.csv' % latent_dim)
