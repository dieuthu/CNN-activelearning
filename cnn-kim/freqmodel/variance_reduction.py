"""Variance Reduction"""
import copy
import numpy as np

from libact.base.interfaces import QueryStrategy
from libact.base.dataset import Dataset
import libact.models
from libact.query_strategies._variance_reduction import estVar
from libact.utils import inherit_docstring_from, zip

from libact.query_strategies.variance_reduction import VarianceReduction



def _Phi(sigma, PI, X, epi, ex, label_count, feature_count):
    ret = estVar(sigma, PI, X, epi, ex)
    return ret


def _E(args):
    X, y, qx, clf, label_count, sigma, model = args
    print ('sigma', sigma)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    query_point = sigmoid(clf.predict_real([qx]))
    feature_count = len(X[0])
    ret = 0.0
    for i in range(label_count):
        clf_ = copy.copy(model)
        clf_.train(Dataset(np.vstack((X, [qx])), np.append(y, i)))
        PI = sigmoid(clf_.predict_real(np.vstack((X, [qx]))))
        ret += query_point[-1][i] * _Phi(sigma, PI[:-1], X, PI[-1], qx,
                                         label_count, feature_count)
    return ret

class MyVarianceReduction(VarianceReduction):
    def __init__(self, *args, **kwargs):
        super(MyVarianceReduction, self).__init__(*args, **kwargs)

    @inherit_docstring_from(VarianceReduction)
    def make_query(self):
        labeled_entries = self.dataset.get_labeled_entries()
        Xlabeled, y = zip(*labeled_entries)
        Xlabeled = np.array(Xlabeled)
        y = list(y)

        unlabeled_entries = self.dataset.get_unlabeled_entries()
        unlabeled_entry_ids, X_pool = zip(*unlabeled_entries)

        label_count = self.dataset.get_num_of_labels()

        clf = copy.copy(self.model)
        clf.train(Dataset(Xlabeled, y))

        #p = Pool(self.n_jobs)
        errors = []
        print('Xpool....', len(X_pool))
        for x in X_pool:
            errors.append(_E([Xlabeled, y, x, clf, label_count, self.sigma, self.model]))
        #p.terminate()

        return unlabeled_entry_ids[errors.index(min(errors))]
