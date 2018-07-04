"""Random Sampling
"""
from libact.base.interfaces import QueryStrategy
from libact.utils import inherit_docstring_from, seed_random_state, zip
from libact.query_strategies import RandomSampling
from sklearn.utils import shuffle

class MyRandomSampling(RandomSampling):
    def __init__(self, *args, **kwargs):
        super(MyRandomSampling, self).__init__(*args, **kwargs)
        self.batch_size = kwargs.pop('batch_size', 10)

    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, _ = zip(*dataset.get_unlabeled_entries())
        unlabeled_entry_ids = shuffle(unlabeled_entry_ids)

        return_len = min(len(unlabeled_entry_ids), self.batch_size)
        entry_id = unlabeled_entry_ids[:return_len]
#        entry_id = [20, 104,106,64,77,37,164,136,127,130]
        return entry_id
        
