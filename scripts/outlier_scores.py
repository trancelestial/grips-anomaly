import numpy as np
from sklearn.decomposition import PCA as pca
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture


class GM:
    def __init__(self, m, **kwargs):
        self.model = GaussianMixture(n_components=m, **kwargs)

    def fit(self, x):
        self.model.fit(x)
        return self

    def score(self, x):
        return self.model.score_samples(x)

    @property
    def name(self):
        return f'GM-{self.model.n_components}'


class PCA:
    def __init__(self, m, **kwargs):
        self.model = pca(n_components=m, **kwargs)

    def fit(self, x):
        self.model.fit(x)
        return self

    def score(self, x):
        return self.model.transform(x)[:, -1]

    @property
    def name(self):
        return f'PC-{self.model.n_components}'


class PCA_RE(PCA):
    def score(self, x):
        z = self.model.transform(x)
        x_reconstruct = self.model.inverse_transform(z)
        norm = np.sqrt(np.square(x - x_reconstruct).sum(axis=1))
        return norm

    @property
    def name(self):
        return f'PCA-RE-{self.model.n_components}'


class ZScore:
    def __init__(self, **kwargs):
        pass

    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

    def score(self, x):
        scores = np.square((x - self.mean) / self.std).sum(axis=1)
        return scores

    @property
    def name(self):
        return 'Z-score'


class IF:
    def __init__(self, **kwargs):
        self.model = IsolationForest(**kwargs)

    def fit(self, x):
        self.model.fit(x)
        return self

    def score(self, x):
        return self.model.score_samples(x)

    @property
    def name(self):
        return 'IF'
