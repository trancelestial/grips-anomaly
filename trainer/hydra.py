from keras.engine.saving import load_model
from sklearn.preprocessing import StandardScaler

from scripts.util import load_credit_card_data
from keras.layers import (
    Input,
    Dense,
)
from trainer.model import ff_relu
from keras import Model
from keras.utils import plot_model
from sklearn.metrics import average_precision_score
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    train, test = load_credit_card_data('../data', mode='train_test')
    train_x = train.drop(columns=['Time', 'Amount', 'Class'])
    train_y = train['Class']
    test_x = test.drop(columns=['Time', 'Amount', 'Class'])
    test_y = test['Class']

    # fit the standard scaler to all train data
    scaler = StandardScaler().fit(train_x)

    # standardize train and test features
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    _, num_features = train_x.shape
    ae_hidden = (64, 32, 16)
    latent_dim = 2
    classifier_hidden = (64, 32)
    alpha = 0.03

    inputs = x = Input((num_features,))
    for units in ae_hidden:
        x = ff_relu(units, alpha=alpha)(x)
    z = x = Dense(latent_dim)(x)
    for units in ae_hidden[::-1]:
        x = ff_relu(units, alpha=alpha)(x)
    ae_outputs = Dense(num_features, name='decoder')(x)
    x = z
    for units in classifier_hidden:
        x = ff_relu(units, alpha=alpha)(x)
    classifier_outputs = Dense(1, activation='sigmoid', name='classifier')(x)

    model = Model(inputs, [ae_outputs, classifier_outputs])

    plot_model(model, show_shapes=True)

    model.compile(optimizer='adam',
                  loss={'decoder': 'mse', 'classifier': 'binary_crossentropy'},
                  metrics={'classifier': 'accuracy', 'decoder': 'mse'})

    model.fit(train_x, {'decoder': train_x, 'classifier': train_y}, batch_size=64,
              epochs=10, verbose=2, validation_split=0.2,
              callbacks=[ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_classifier_acc', verbose=1)])

    model = load_model('best_model.h5')

    classifier = Model(inputs, classifier_outputs)
    encoder = Model(inputs, ae_outputs)

    test_xhat = encoder.predict(test_x)
    test_yhat = classifier.predict(test_x)

    auc_pr = average_precision_score(test_y, test_yhat)

    print(auc_pr)

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
        plt.savefig('test_embedding_space.png')
