from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Input,
    Dense,
)
from keras.layers import LeakyReLU, BatchNormalization
from keras.models import load_model
from keras.utils import plot_model
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scripts.util import load_credit_card_data
from trainer.model import ff_block

if __name__ == '__main__':
    train, test = load_credit_card_data('../data', mode='train_test')
    train_x = train.drop(columns=['Time', 'Amount', 'Class'])
    train_y = train['Class']
    test_x = test.drop(columns=['Time', 'Amount', 'Class'])
    test_y = test['Class']

    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, stratify=train_y, random_state=12345,
                                                      test_size=0.2)

    # fit the standard scaler to all train data
    scaler = StandardScaler().fit(train_x)

    # standardize train and test features
    train_x_norm = scaler.transform(train_x)
    test_x_norm = scaler.transform(test_x)
    val_x_norm = scaler.transform(val_x)

    _, num_features = train_x.shape
    hidden = (32,) * 4
    alpha = 0.2
    bn = False

    inputs = x = Input((num_features,))
    x = Dense(4 * hidden[0])(x)
    x_shortcut = None
    for i, units in enumerate(hidden[1:]):
        x = ff_block(units, alpha=alpha, bottleneck_factor=4, bn=bn)(x)
    if bn:
        x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    plot_model(model, show_shapes=True, show_layer_names=False)
    model.summary()

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_x_norm, train_y, batch_size=64,
              callbacks=[ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_acc', verbose=1)],
              epochs=20, verbose=2, validation_data=(val_x_norm, val_y))

    model = load_model('best_model.h5')
    test_yhat = model.predict(test_x_norm).flatten()

    auc_pr = average_precision_score(test_y, test_yhat)

    print(f'Test AUC-PR: {auc_pr:0.5f}')
