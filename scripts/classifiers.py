import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import train_test_split

from scripts.outlier_scores import *
from scripts.util import load_credit_card_data

SEED = 123
NUM_TESTS = 20

if __name__ == '__main__':
    np.random.seed(SEED)

    train, test = load_credit_card_data('../data', mode='train_test')

    df_train_x = train.drop(columns=['Time', 'Class', 'Amount'])
    df_train_y = train['Class']

    df_test_x = test.drop(columns=['Time', 'Class', 'Amount'])
    df_test_y = test['Class']

    outlier_model_combinations = [
        [],
        [(ZScore, {})],
        [(PCA, {'m': 1, 'whiten': True})],
        [(PCA_RE, {'m': 1, 'whiten': True})],
        [(IF, {})],
        [(GM, {'m': 1})],
        [
            (ZScore, {}),
            (PCA, {'m': 1, 'whiten': True}),
            (PCA_RE, {'m': 1, 'whiten': True}),
            (IF, {}),
            (GM, {'m': 1})
        ]
    ]

    seeds = np.random.randint(low=np.iinfo(np.int32).max, size=NUM_TESTS)
    history = []

    for outlier_models_signatures in outlier_model_combinations:
        outlier_models_names = [model(**params).name for model, params in outlier_models_signatures]

        print(f'current outlier models: {outlier_models_names}')

        performance_scores = []

        for i, seed in enumerate(seeds):
            outlier_models = [outlier_model(**{**params, 'random_state': seed})
                              for outlier_model, params in outlier_models_signatures]

            train_x, val_x, train_y, val_y = train_test_split(df_train_x, df_train_y,
                                                              stratify=df_train_y,
                                                              random_state=seed,
                                                              test_size=0.2)

            train_outlier_scores_data = {}
            val_outlier_scores_data = {}
            test_outlier_scores_data = {}

            for outlier_model in outlier_models:
                # fit the outlier model on only the non-fraudulent transactions
                outlier_model.fit(train_x.loc[train_y == 0])

                train_outlier_scores = outlier_model.score(train_x)
                val_outlier_scores = outlier_model.score(val_x)
                test_outlier_scores = outlier_model.score(df_test_x)

                train_outlier_scores_data[outlier_model.name] = train_outlier_scores
                val_outlier_scores_data[outlier_model.name] = val_outlier_scores
                test_outlier_scores_data[outlier_model.name] = test_outlier_scores

            train_x_with_scores = train_x.join(pd.DataFrame(train_outlier_scores_data))
            val_x_with_scores = val_x.join(pd.DataFrame(val_outlier_scores_data))
            test_x_with_scores = df_test_x.join(pd.DataFrame(test_outlier_scores_data))

            classifier = xgb.XGBClassifier(n_estimators=100,
                                           max_depth=5,
                                           eta=0.1,
                                           random_state=seed,)

            classifier.fit(train_x_with_scores, train_y,
                           eval_set=[(val_x_with_scores, val_y)],
                           eval_metric='error',
                           verbose=False,)

            pred_test_y = classifier.predict_proba(test_x_with_scores)[:, 1]

            auc_pr = metrics.average_precision_score(df_test_y, pred_test_y)
            auc_roc = metrics.roc_auc_score(df_test_y, pred_test_y)

            performance_scores.append({
                'auc_pr': auc_pr,
                'auc_roc': auc_roc
            })

            history.append([
                '|'.join(outlier_models_names) or 'baseline',
                i + 1,
                auc_pr,
                auc_roc,
                seed
            ])

            print(f'[Iteration {i + 1}/{NUM_TESTS}] AUC-PR: {auc_pr:0.4f} AUC-ROC: {auc_roc:0.4f}')

        auc_pr_scores = [score['auc_pr'] for score in performance_scores]
        mean_auc_pr = np.mean(auc_pr_scores)
        auc_pr_std = np.std(auc_pr_scores)

        print(f'avg AUC-PR: {mean_auc_pr} (\u00B1{auc_pr_std})')

    history = pd.DataFrame(data=history,
                           columns=['outlier_models', 'iteration', 'auc_pr', 'auc_roc', 'seed'])

    history.to_csv('history_xgb.csv', index=False, sep=';')
