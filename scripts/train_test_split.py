import os
import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    SEED = 12345

    if not os.path.isdir('../data'):
        raise ValueError('Put \'creditcard.csv\' into \'data\' in the root directory')

    df = pd.read_csv('../data/creditcard.csv')

    train, test = train_test_split(df, test_size=0.25, stratify=df.Class.values, random_state=SEED)

    train.to_csv('../data/creditcard_train.csv', index=False)
    test.to_csv('../data/creditcard_test.csv', index=False)
