import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from scipy.stats import wasserstein_distance
from sklearn import utils
from sklearn.preprocessing import StandardScaler

from scripts.util import load_credit_card_data
from trainer.model import ff_relu


def generator(num_features, latent_dim, hidden, alpha=0., bn=None):
    inputs = x = Input((latent_dim,))
    for units in hidden:
        x = ff_relu(units, alpha=alpha, bn=bn)(x)
    outputs = Dense(num_features)(x)
    model = Model(inputs, outputs, name='generator')
    return model


def discriminator(num_features, hidden, alpha=0., bn=None):
    inputs = x = Input((num_features,))
    for units in hidden:
        x = ff_relu(units, alpha=alpha, bn=bn)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs, name='discriminator')
    return model


class GAN:
    def __init__(self, num_features, latent_dim, *,
                 generator_kwargs, discriminator_kwargs, optimizer=None):
        self.num_features = num_features
        self.latent_dim = latent_dim

        if optimizer is None:
            optimizer = Adam()

        # Build and compile the discriminator
        self.discriminator = discriminator(num_features, **discriminator_kwargs)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = generator(num_features, latent_dim, **generator_kwargs)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        generator_in = self.generator.input
        validity = self.discriminator(self.generator(generator_in))

        # The combined model (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(generator_in, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def train(self, x, epochs=1, k=None, batch_size=32, verbose=1, shuffle=True, cb=None):
        if k is None:
            k = lambda _: 1
        elif isinstance(k, int):
            k_ = k
            k = lambda _: k_
        elif not callable(k):
            raise ValueError('k must be a schedule function or int')

        num_examples, *_ = x.shape
        num_batches = num_examples // batch_size
        idx = np.arange(num_examples)
        y_valid = np.ones((batch_size, 1))
        y_fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(idx)

            k_i = k(epoch)
            d_losses = []
            g_losses = []

            for batch_idx, batch_end in enumerate(range(batch_size, num_examples + 1, batch_size), start=1):
                batch_start = batch_end - batch_size

                # train discriminator
                x_batch = x[idx[batch_start:batch_end]]
                x_noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                x_generated = self.generator.predict(x_noise)

                x_d = np.concatenate([x_batch, x_generated])
                y_d = np.concatenate([y_valid, y_fake])
                x_d, y_d = utils.shuffle(x_d, y_d)

                d_loss = self.discriminator.train_on_batch(x_d, y_d)
                d_losses.append(d_loss)

                # train generator
                if batch_idx % k_i == 0:
                    x_noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                    g_loss = self.combined.train_on_batch(x_noise, y_valid)
                    g_losses.append(g_loss)

                    if verbose == 1:
                        print(f'Batch {batch_idx}/{num_batches} '
                              f'[D loss: {d_loss[0]:0.3f} acc: {d_loss[1]:.2%}] '
                              f'[G loss: {g_loss:.3f}]')

            avg_d_loss, avg_d_acc = np.mean(d_losses, axis=0)
            avg_g_loss = np.mean(g_losses, axis=0)

            if verbose:
                print(f'Epoch {epoch + 1}/{epochs} '
                      f'[D loss: {avg_d_loss:0.3f} acc: {avg_d_acc:.2%}] '
                      f'[G loss: {avg_g_loss:.3f}]')

            if cb is not None:
                cb(epoch)

    def sample_data(self, size):
        x_noise = np.random.normal(0, 1, (size, self.latent_dim))
        x_generated = self.generator.predict(x_noise)
        return x_generated

    def load_weights(self, filepath):
        with open(filepath, 'rb') as handle:
            weights = pickle.load(handle)
            self.combined.set_weights(weights['model'])
            self.discriminator.optimizer.set_weights(weights['optimizer'])
            self.combined.optimizer.set_weights(weights['optimizer'])

        return self

    def save(self, filepath):
        weights = {
            'model': self.combined.get_weights(),
            'optimizer': self.combined.optimizer.get_weights(),
        }

        with open(filepath, 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    os.makedirs('gan', exist_ok=True)

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

    gan = GAN(num_features=num_features,
              latent_dim=latent_dim,
              generator_kwargs={'hidden': (128, 128, 128, 128, 128), 'alpha': 0.2, 'bn': {'momentum': 0.8}},
              discriminator_kwargs={'hidden': (32, 16), 'alpha': 0.2},
              optimizer=Adam(0.0002, 0.5))

    gan.generator.summary()
    gan.discriminator.summary()
    gan.combined.summary()

    def k_schedule(epoch):
        if epoch:
            return 1
        else:
            return 10

    def checkpoint(epoch):
        gan.save(f'gan/gan_{epoch}.h5')

    gan.train(train_x, epochs=20, k=k_schedule, cb=checkpoint)

    x_gen = gan.sample_data(size=len(test_x))
    x_gen = scaler.inverse_transform(x_gen)

    cols = int(np.sqrt(num_features))
    rows = int(np.ceil(num_features / cols))

    f, axes = plt.subplots(rows, cols, figsize=(4 * rows, 4 * cols))

    for i, col in enumerate(test_x_.columns[:num_features]):
        print(f'plotting feature distribution: {col}')

        col_i = i % cols
        row_i = i // cols

        train_emd = wasserstein_distance(train_x_[col], x_gen[:, i])
        test_emd = wasserstein_distance(test_x_[col], x_gen[:, i])

        sns.distplot(train_x_[col], ax=axes[row_i, col_i], hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3},
                     label=f'train (n={len(train_x)}) [EMD={train_emd:0.4f}]')

        sns.distplot(test_x_[col], ax=axes[row_i, col_i], hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3},
                     label=f'test (n={len(test_x)}) [EMD={test_emd:0.4f}]')

        sns.distplot(x_gen[:, i], ax=axes[row_i, col_i], hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3},
                     label=f'gen (n={len(x_gen)})')

    plt.tight_layout()
    plt.savefig('gan/gan_features_norm+fraud.png')
