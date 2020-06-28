import pandas as pd
import os


def load_credit_card_data(dir, mode='original'):
    if mode == 'original':
        file = os.path.join(dir, 'creditcard.csv')
        data = pd.read_csv(file)
    elif mode == 'train_test':
        train_file = os.path.join(dir, 'creditcard_train.csv')
        test_file = os.path.join(dir, 'creditcard_test.csv')
        train = pd.read_csv(train_file)
        test = pd.read_csv(test_file)
        data = train, test
    else:
        raise ValueError('unknown mode: ' + mode)
    return data
