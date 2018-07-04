""" Uncertainty Sampling

This module contains a class that implements two of the most well-known
uncertainty sampling query strategies: the least confidence method and the
smallest margin method (margin sampling).

"""
import numpy as np

from libact.base.interfaces import QueryStrategy, ContinuousModel, \
    ProbabilisticModel
from libact.utils import inherit_docstring_from, zip
from libact.query_strategies.uncertainty_sampling import UncertaintySampling

class MyUncertaintySampling(UncertaintySampling):
    def __init__(self, *args, **kwargs):
        super(MyUncertaintySampling, self).__init__(*args, **kwargs)
        self.batch_size = kwargs.pop('batch_size', 10)

    def make_query(self):
        askid, result = super(MyUncertaintySampling, self).make_query(return_score=True)
        sorted_ids = [t[0] for t in sorted(result, key=lambda x:x[1], reverse=True)]
        returnlen = min(len(sorted_ids), self.batch_size)
        return sorted_ids[:returnlen]

