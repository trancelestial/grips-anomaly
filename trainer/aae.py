import os
import pickle

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Input, Dense, Lambda, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from scipy.stats import wasserstein_distance, truncnorm
from sklearn import utils
from sklearn.preprocessing import StandardScaler

from scripts.util import load_credit_card_data
from trainer.model import ff_relu


def generator(num_features, latent_dim, hidden, dense_kwargs=None,
              encoder='deterministic', num_classes=None):
    if dense_kwargs is None:
        dense_kwargs = {}

    encoder_inputs = x = Input((num_features,))

    for units in hidden:
        x = ff_relu(units, **dense_kwargs)(x)

    if encoder == 'gaussian':
        mu = Dense(latent_dim, name='mu')(x)
        log_var = Dense(latent_dim, name='log_var')(x)
        z = Lambda(lambda x: x[0] + K.random_normal(K.shape(x[0])) * K.exp(x[1] / 2),
                   output_shape=lambda x: x[0], name='z')([mu, log_var])
    elif encoder == 'deterministic':
        z = Dense(latent_dim)(x)
    else:
        raise ValueError('unknown encoder type: %s' % encoder)

    if num_classes is None:
        decoder_inputs = x = Input((latent_dim,))
    else:
        decoder_inputs = [Input((latent_dim,)), Input((num_classes,))]
        x = Concatenate()(decoder_inputs)

    for units in hidden[::-1]:
        x = ff_relu(units, **dense_kwargs)(x)

    outputs = Dense(num_features)(x)

    encoder = Model(encoder_inputs, z, name='encoder')
    decoder = Model(decoder_inputs, outputs, name='decoder')

    return encoder, decoder


def discriminator(latent_dim, hidden, dense_kwargs=None):
    if dense_kwargs is None:
        dense_kwargs = {}

    inputs = x = Input((latent_dim,))

    for units in hidden:
        x = ff_relu(units, **dense_kwargs)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='discriminator')

    return model


class AAE:
    def __init__(self, num_features, latent_dim, *,
                 generator_kwargs, discriminator_kwargs,
                 optimizer=None, num_classes=None):
        if optimizer is None:
            optimizer = Adam()

        # Build and compile the discriminator
        self.discriminator = discriminator(latent_dim, **discriminator_kwargs)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.encoder, self.decoder = generator(num_features, latent_dim, **generator_kwargs,
                                               num_classes=num_classes,)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        inputs = self.encoder.input
        z = self.encoder(inputs)
        validity = self.discriminator(z)

        if num_classes is None:
            ae = self.decoder(z)
        else:
            _, class_inputs = self.decoder.input
            ae = self.decoder([z, class_inputs])
            inputs = [inputs, class_inputs]

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(inputs, [ae, validity])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
                              loss_weights=[0.99, 0.01],
                              optimizer=optimizer)

    def train(self, x, y=None, epochs=1, k=None, batch_size=128, verbose=1, shuffle=True, cb=None):
        if k is None:
            k = lambda _: 1
        elif isinstance(k, int):
            k_ = k
            k = lambda _: k_
        elif not callable(k):
            raise ValueError('k must be a schedule function or int')

        _, latent_dim = self.encoder.output_shape
        num_examples, *_ = x.shape
        num_batches = num_examples // batch_size
        idx = np.arange(num_examples)
        y_valid = np.ones((batch_size, 1))
        y_fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(idx)

            k_epoch = k(epoch)
            d_losses = []
            g_losses = []

            for i, batch_end in enumerate(range(batch_size, num_examples + 1, batch_size), start=1):
                batch_start = batch_end - batch_size
                batch_idx = idx[batch_start:batch_end]

                # train discriminator
                x_batch = x[batch_idx]
                latent_real = np.random.normal(0, 1, (batch_size, latent_dim))
                latent_fake = self.encoder.predict(x_batch)

                x_d = np.concatenate([latent_real, latent_fake])
                y_d = np.concatenate([y_valid, y_fake])
                x_d, y_d = utils.shuffle(x_d, y_d)

                d_loss = self.discriminator.train_on_batch(x_d, y_d)
                d_losses.append(d_loss)

                # train generator
                if i % k_epoch == 0:
                    if y is not None:
                        y_batch = y[batch_idx]
                        x_g = [x_batch, y_batch]
                    else:
                        x_g = x_batch
                    g_loss = self.combined.train_on_batch(x_g, [x_batch, y_valid])
                    g_losses.append(g_loss)

                    if verbose == 1:
                        print(f'Batch {i}/{num_batches} '
                              f'[D loss: {d_loss[0]:0.3f} acc: {d_loss[1]:.2%}] '
                              f'[G loss: {g_loss[1]:.3f} mse: {g_loss[0]:.3f}]')

            avg_d_loss, avg_d_acc = np.mean(d_losses, axis=0)
            avg_g_loss = np.mean(g_losses, axis=0)

            if verbose:
                print(f'Epoch {epoch + 1}/{epochs} '
                      f'[D loss: {avg_d_loss:0.3f} acc: {avg_d_acc:.2%}] '
                      f'[G loss: {avg_g_loss[1]:.3f} mse: {avg_g_loss[0]:.3f}]')

            if cb is not None:
                cb(epoch)

    def sample_data(self, y=None, size=1, x_noise=None):
        if x_noise is None:
            _, latent_dim = self.encoder.output_shape
            x_noise = np.random.normal(0, 1, (size, latent_dim))
        if y is None:
            x = x_noise
        else:
            x = [x_noise, y]
        x_generated = self.decoder.predict(x)
        return x_generated

    def load_weights(self, filepath):
        with open(filepath, 'rb') as handle:
            weights = pickle.load(handle)
            self.combined.set_weights(weights['model'])
            self.combined.optimizer.set_weights(weights['optimizer'])

        return self

    def save(self, filepath):
        weights = {
            'model': self.combined.get_weights(),
            'optimizer': self.combined.optimizer.get_weights(),
        }

        with open(filepath, 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)


def truncated_normal(size, mu=0., sigma=1.):
    lower = mu - 2 * sigma
    upper = mu + 2 * sigma
    n = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    return n.rvs(size=size)


if __name__ == '__main__':
    job_dir = 'aae'
    current_epoch = 0
    epochs = 50

    os.makedirs(job_dir, exist_ok=True)

    train, test = load_credit_card_data('../data', mode='train_test')

    train_x = train_x_ = train.drop(columns=['Time', 'Amount', 'Class'])
    train_y = train['Class']
    test_x = test_x_ = test.drop(columns=['Time', 'Amount', 'Class'])
    test_y = test['Class']

    num_features = len(train_x.columns)
    latent_dim = 8

    # fit the standard scaler to all train data
    scaler = StandardScaler().fit(train_x)

    # standardize train and test features
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    gan = AAE(num_features=num_features,
              latent_dim=latent_dim,
              generator_kwargs={'hidden': (128, 128),
                                'dense_kwargs': {'alpha': 0.2},
                                'encoder': 'deterministic'},
              discriminator_kwargs={'hidden': (64, 32),
                                    'dense_kwargs': {'alpha': 0.2}},
              optimizer=Adam(0.0002, 0.5),
              num_classes=1)

    plot_model(gan.combined, f'{job_dir}/combined.png', show_layer_names=True, show_shapes=True)
    plot_model(gan.encoder, f'{job_dir}/encoder.png', show_layer_names=True, show_shapes=True)
    plot_model(gan.decoder, f'{job_dir}/decoder.png', show_layer_names=True, show_shapes=True)
    plot_model(gan.discriminator, f'{job_dir}/discriminator.png', show_layer_names=True, show_shapes=True)

    if current_epoch > 0:
        # FIXME: hack that forces keras to initialize the optimizer
        #  in order to update its weights from the checkpoint
        gan.train(train_x[:128], train_y[:128])
        gan.load_weights(f'{job_dir}/aae_{current_epoch}.p')

    def checkpoint(epoch):
        gan.save(f'{job_dir}/aae_{epoch + current_epoch + 1}.p')

    gan.train(train_x, train_y, epochs=epochs, k=1, cb=checkpoint, verbose=2)

    noise = truncated_normal(size=(len(test_x), latent_dim))
    x_gen_fraud = gan.sample_data(y=np.ones(len(test_x)), x_noise=noise)
    x_gen_norm = gan.sample_data(y=np.zeros(len(test_x)), x_noise=noise)

    x_gen_fraud = scaler.inverse_transform(x_gen_fraud)
    x_gen_norm = scaler.inverse_transform(x_gen_norm)

    pd.DataFrame(data=x_gen_fraud, columns=train_x_.columns).to_csv(
        f'{job_dir}/gen_fraud_{current_epoch + epochs}.csv', index=False)
    pd.DataFrame(data=x_gen_norm, columns=train_x_.columns).to_csv(
        f'{job_dir}/gen_norm_{current_epoch + epochs}.csv', index=False)

    cols = int(np.sqrt(num_features))
    rows = int(np.ceil(num_features / cols))

    f, axes = plt.subplots(rows, cols, figsize=(4 * rows, 4 * cols))

    for i, col in enumerate(test_x_.columns[:num_features]):
        print(f'plotting feature distribution: {col}')

        col_i = i % cols
        row_i = i // cols

        train_emd = wasserstein_distance(train_x_[col], x_gen_fraud[:, i])
        test_emd = wasserstein_distance(test_x_[col], x_gen_fraud[:, i])

        sns.distplot(train_x_[col], ax=axes[row_i, col_i], hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3},
                     label=f'train (n={len(train_x)}) [EMD={train_emd:0.4f}]')

        sns.distplot(test_x_[col], ax=axes[row_i, col_i], hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3},
                     label=f'test (n={len(test_x)}) [EMD={test_emd:0.4f}]')

        sns.distplot(x_gen_fraud[:, i], ax=axes[row_i, col_i], hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3},
                     label=f'gen (n={len(x_gen_fraud)})')

    plt.tight_layout()
    plt.savefig(f'{job_dir}/gan_features_norm+fraud_{current_epoch + epochs}.png')
