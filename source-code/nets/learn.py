import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from collections import namedtuple

EvaluationResult = namedtuple('EvaluationResult', ['q2','r2'])
def evaluate(predicted, real):
    '''Evaluate by computing both squared pearson correlation and q2 score'''
    error = predicted - real
    base = real - real.mean()
    q2 = 1. - np.dot(error,error)/np.dot(base,base)
    r2 = np.corrcoef(predicted, real)[0,1]**2
    return EvaluationResult(q2=q2, r2=r2)



RegionResult = namedtuple('RegionResult', ['partials', 'final'])
class RegionRegressor(object):
    '''Region regression estimator

    regressor : estimator
        Regression object to use for region-density estimation (default is
        RandomForestRegressor)
    '''
    def __init__(self,
                random_state=None,
                n_estimators=100,
                n_jobs=1,
                regressor=None,
                ):
        if regressor is None:
            self.regressor = RandomForestRegressor(random_state=random_state,
                n_estimators=n_estimators, n_jobs=n_jobs, oob_score=True)
        else:
            self.regressor = regressor
        self.lr = LinearRegression()

    def fit(self, features, labels, fractions):
        ffeatures = features
        features = np.concatenate(features)
        features = features.astype(float)
        labels = np.concatenate(labels)
        labels = labels.astype(float)
        features = np.nan_to_num(features)
        self.regressor.fit(features, labels)
        raw = []

        # Out-of-band prediction is better, so use it if possible:
        if hasattr(self.regressor, 'oob_prediction_'):
            next = 0
            for f in ffeatures:
                n = len(f)
                s = f.T[0]
                raw.append(np.dot(self.regressor.oob_prediction_[next:next+n],s)/s.sum())
                next += n
        else:
            for f in ffeatures:
                s = f.T[0]
                raw.append(np.dot(self.regressor.predict(f),s)/s.sum())
        self.raw_ = np.array(raw)
        self.lr = None
        if fractions is not None:
            self.lr = LinearRegression()
            self.lr.fit(np.atleast_2d(self.raw_).T, fractions)
        return self

    def predict(self, fs, return_partials=False, fix_line=True, clip01=True):
        '''
        clip01 : boolean, optional
            Whether to clip results to 0-1 range. Defaults to True.
        '''
        vs = self.regressor.predict(np.nan_to_num(fs.astype(float)))
        s = fs.T[0]
        final = np.dot(vs, s)/s.sum()
        if fix_line:
            assert self.lr is not None
            final = self.lr.predict(final)
        if clip01:
            final = np.clip(final, 0, 1.)
        if return_partials:
            return RegionResult(vs, final)
        return final


class Combined(object):
    def __init__(self, mode='matrix',
                random_state=None,
                n_estimators=100,
                n_jobs=1):
        if mode not in {'matrix', 'average'}:
            raise ValueError('Unknown mode "{}".'.format(mode))
        self.mode = mode
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def fit(self, features_labels, fractions):
        self.base = []
        n = len(features_labels[0])
        raws = []
        for ifeat in range(n):
            r = RegionRegressor(random_state=self.random_state, n_estimators=self.n_estimators, n_jobs=self.n_jobs)
            features = [f[ifeat][0] for f in features_labels]
            labels = [f[ifeat][1] for f in features_labels]
            r.fit(features, labels, None)
            self.base.append(r)
            raws.append(r.raw_)
        self.lr = LinearRegression()
        if self.mode == 'matrix':
            self.lr.fit(np.array(raws).T, fractions)
        else:
            avgs = np.array(raws).sum(0)
            self.lr.fit(np.atleast_2d(avgs).T, fractions)
        return self

    def predict(self, features, clip01=True):
        assert len(features) == len(self.base)
        raw = np.array([b.predict(fs, fix_line=False) for fs,b in zip(features, self.base)])
        if self.mode == 'average':
            raw = raw.sum()
        pred = self.lr.predict(np.atleast_2d(raw))
        if clip01:
            pred = np.clip(pred, 0., 1.0)
        return pred

