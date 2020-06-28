import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()


if __name__ == '__main__':
    h = pd.read_csv('history_xgb.csv', sep=';')

    for m, results in h.groupby('outlier_models'):
        print(m, 'AUC-PR: {avg:0.3f} (\u00b1{std:0.3f})'.format(
            avg=results.auc_pr.mean(),
            std=results.auc_pr.std()))

    ax = sns.catplot(x='outlier_models', y='auc_pr', data=h, kind='box')
    ax.set_xticklabels(rotation=90)

    plt.savefig('history_xgb.png')
