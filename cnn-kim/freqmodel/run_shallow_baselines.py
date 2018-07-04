from sklearn.linear_model import LogisticRegression
from datautil import load_eng_all_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

from datautil import load_zho_tfidplus_features, load_eng_all_features, get_set_random_states, new_result_item
# >>> X = [[0, 0], [1, 1]]
# >>> Y = [0, 1]
# >>> clf = RandomForestClassifier(n_estimators=10)
# >>> clf = clf.fit(X, Y)

def split_data(X,y, nfold, foldid, random_state=0):
    """
    foldid = 0,1,2,..., nfold-1
    """
    foldid = foldid + 1
    X, y = shuffle(X,y, random_state=random_state)

    nsamples =  len(y)
    nsample_per_fold = nsamples//nfold
    idx = range(len(y))


    if foldid != nfold:
        test_idx = list(range((foldid-1)*nsample_per_fold, foldid*nsample_per_fold))
    else: test_idx = list(range((foldid-1)*nsample_per_fold, nsamples))

    train_idx = [i for i in idx if i not in test_idx]

    train_size = len(train_idx)
    dev_size = train_size//10

    test_X = X[test_idx]
    test_y = y[test_idx]
    dev_X = X[train_idx[0:dev_size],:]
    dev_y = y[train_idx[0:dev_size]]
    train_X = X[train_idx[dev_size:],:]
    train_y = y[train_idx[dev_size::]]
    return train_X, train_y,  test_X, test_y, dev_X, dev_y

import itertools
def test_lr(X,y, output_file):
    result = []
    nfold = 3
    for i,j in itertools.product(get_set_random_states(),range(nfold)):
        train_data, y_train, test_data, y_test, dev_X, dev_y = split_data(X,y, nfold, j, random_state=i)
        
        clf = LogisticRegression()
        clf.fit(train_data, y_train)
        y_pred=clf.predict(test_data)
        f1=f1_score(y_test, y_pred, average='macro')
        result.append(new_result_item('state={0}/foldid={1}'.format(i,j), 'lr', None, len(y_train), f1))

    name = ['experimen_no', 'algorithm', 'query_strategy', 'number_of_instances', 'f1']
    result = pd.DataFrame(columns=name, data=result)
    result.to_csv(output_file)


def test_random_forest(X,y, output_file):
    result = []
    nfold = 3
    for i,j in itertools.product(get_set_random_states(),range(nfold)):
        train_data, y_train, test_data, y_test, dev_X, dev_y = split_data(X,y, nfold, j, random_state=i)
                
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(train_data, y_train)
        y_pred=clf.predict(test_data)

        f1=f1_score(y_test, y_pred, average='macro')
        result.append(new_result_item('state={0}/foldid={1}'.format(i,j), 'random-forest', None, len(y_train), f1))

    name = ['experimen_no', 'algorithm', 'query_strategy', 'number_of_instances', 'f1']
    result = pd.DataFrame(columns=name, data=result)
    result.to_csv(output_file)

def test_ada_boost(X,y):
 #   X, y = load_eng_all_features()
    train_data, test_data, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    clf = AdaBoostClassifier(n_estimators=10)
    clf.fit(train_data, y_train)
    y_pred=clf.predict(test_data)
    print ('adaboost, f1 scores:', f1_score(y_test, y_pred, average='macro'))

if __name__=='__main__':
#    print ('test english')
#    X, y = load_eng_all_features()

#    test_lr(X,y)
#    test_random_forest(X,y)
#    test_ada_boost(X,y)

    print ('test chinese')
    X, y = load_zho_tfidplus_features()
    y = np.ravel(y)

    test_lr(X,y, '../../datalight/result/lr_zho_baseline.csv')
    test_random_forest(X,y, '../../datalight/result/rf_zho_baseline.csv')

