import os

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Input,
    Dense,
    ReLU,
    Lambda,
    BatchNormalization,
)
from sklearn.preprocessing import StandardScaler

from scripts.util import load_credit_card_data


def get_most_recent_file(directory):
    files = (os.path.join(directory, file) for file in os.listdir(directory))
    return max(files, key=os.path.getctime)


def to_dataset(creditcard_data):
    x = creditcard_data.drop(columns=['Time', 'Class', 'Amount'])
    # x['Amount'] = np.log(1 + creditcard_data.Amount)
    y = creditcard_data['Class']
    return x, y


def sampling(x):
    z_mu, z_log_var = x
    batch_size = K.shape(z_mu)[0]
    _, latent_dim = K.int_shape(z_mu)
    eps = K.random_normal(shape=(batch_size, latent_dim))
    return z_mu + K.exp(0.5 * z_log_var) * eps


def vae_loss(y_true, y_pred):
    # reconstruction_loss = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    reconstruction_loss = K.sum(K.square(y_true - y_pred), axis=-1)
    kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mu) - 1 - z_log_var, axis=1)
    return reconstruction_loss + beta * kl_loss


def ff_relu(units, bn=False):
    def f(x):
        x = Dense(units)(x)
        if bn:
            x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
    return f


if __name__ == '__main__':
    TRAIN = True
    seed = 12345

    np.random.seed(seed)
    tf.set_random_seed(seed)

    train, test = load_credit_card_data('../data', mode='train_test')

    train_x, train_y = to_dataset(train)
    test_x, test_y = to_dataset(test)

    # fit the standard scaler to all train data
    scaler = StandardScaler().fit(train_x)

    # standardize train and test features
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    _, num_features = train_x.shape
    latent_dim = 16
    beta = 1.0
    hidden = (200, 100)
    job_dir = 'vae_%dd' % latent_dim

    in_ = x = Input(shape=(num_features,))
    for units in hidden:
        x = ff_relu(units, bn=False)(x)
    z_mu = Dense(latent_dim, name='z_mu')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = x = Lambda(sampling, name='z')([z_mu, z_log_var])
    for units in hidden[::-1]:
        x = ff_relu(units, bn=False)(x)
    out_ = Dense(num_features, activation='linear')(x)
    vae = Model(in_, out_)

    vae.compile(optimizer='adam', loss=vae_loss, metrics=['mse'])

    vae.summary()

    if TRAIN:
        # create job directory
        os.makedirs(job_dir, exist_ok=True)

        # define checkpoint saving the currently best model after each epoch
        checkpoint = ModelCheckpoint(job_dir + '/model_{epoch}_{val_loss:0.4f}', save_best_only=True)

        # fit the vae to all train data
        vae.fit(train_x, train_x, validation_split=0.2, batch_size=64, epochs=50, verbose=2, callbacks=[checkpoint])

        # get the most recent file in the job directory which happens to be the last best model
        best_model_file = get_most_recent_file(job_dir)

        # save current model so that the training can be resumed later
        vae.save(job_dir + '/model.h5')
    else:
        # get the most recent file in the job directory which happens to be the last best model
        best_model_file = get_most_recent_file(job_dir)

    # load weights from the best model
    vae.load_weights(best_model_file)
    print('loaded weights from', best_model_file)

    encoder = Model(in_, z_mu)

    train_xhat = encoder.predict(train_x)
    test_xhat = encoder.predict(test_x)

    if latent_dim in [1, 2]:
        test_xhat_nonfraud = test_xhat[test_y == 0]
        test_xhat_fraud = test_xhat[test_y == 1]
        if latent_dim == 1:
            plt.scatter(test_xhat_nonfraud, np.zeros_like(test_xhat_nonfraud), color='b', alpha=0.25, label='nonfraud')
            plt.scatter(test_xhat_fraud, np.zeros_like(test_xhat_fraud), color='r', alpha=0.5, marker='*', label='fraud')
        elif latent_dim == 2:
            plt.scatter(test_xhat_nonfraud[:, 0], test_xhat_nonfraud[:, 1], color='b', alpha=0.25, label='nonfraud')
            plt.scatter(test_xhat_fraud[:, 0], test_xhat_fraud[:, 1], color='r', alpha=0.5, marker='*', label='fraud')
        plt.legend()
        plt.title('Test embedding space')
        plt.savefig(job_dir + '/test_embedding_space.png')

    pd.DataFrame(data=train_xhat,
                 index=train.index,
                 columns=['E%d' % (i + 1) for i in range(latent_dim)]).to_csv(
        job_dir + '/train_embedding_%dd.csv' % latent_dim)

    pd.DataFrame(data=test_xhat,
                 index=test.index,
                 columns=['E%d' % (i + 1) for i in range(latent_dim)]).to_csv(
        job_dir + '/test_embedding_%dd.csv' % latent_dim)
