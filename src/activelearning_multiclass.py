import pdb
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from datautil import load_eng_all_features
from sklearn.model_selection import train_test_split

from libact.base.dataset import Dataset
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler

from sklearn.svm import SVR
from libact.query_strategies.multiclass import ActiveLearningWithCostEmbedding as ALCE

def run(trn_ds, tst_ds, lbr, model, qs, quota):
    E_in, E_out, fc = [], [], []

    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

        X_tst, y_tst = tst_ds.format_sklearn()
        y_pred=model.model.predict(X_tst)
        fc=np.append(fc, f1_score(y_tst, y_pred, average='macro'))
    return E_in, E_out, fc

def split_train_test(X, y, test_size, n_labeled, random_state=0):
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=random_state)

    # make sure there is at least one sample of one class
    classes = np.unique(y_train)
    n_sample_perclass = (int)(n_labeled/len(classes))
    n_labeled = n_sample_perclass*len(classes)
    
    seed = []
    idxes = []
    for l in classes:
        idx = [int(i) for i in range(len(y_train)) if y_train[i]==l]

        idx = shuffle(idx, random_state=random_state)
        
        seed = np.concatenate((seed,idx[:n_sample_perclass]),0)
        idxes = np.concatenate((idxes,idx[n_sample_perclass:]),0)

    idxes = shuffle(idxes, random_state=random_state)
    idxes = np.array(np.concatenate((seed,idxes),0),dtype=int)
        
    X_train = X_train[idxes]
    y_train = y_train[idxes]

    n_labeled = n_sample_perclass*len(classes)
    
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds, n_labeled


def run(trn_ds, tst_ds, lbr, model, qs, quota):
    E_in, E_out, fc = [], [], []

    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

        X_tst, y_tst = tst_ds.format_sklearn()
        y_pred=model.model.predict(X_tst)
        fc=np.append(fc, f1_score(y_tst, y_pred, average='macro'))
    return E_in, E_out, fc

if __name__=='__main__':
    X, y = load_eng_all_features()

    # to make the labels from 0 to 3 instead of 1 to 4
    for i in range(len(y)):
        y[i] = y[i] - 1
        
    n_labeled=12
    test_size=0.3
    n_classes = 4
    
    # Load dataset
    trn_ds, tst_ds, y_train, fully_labeled_trn_ds, n_labeled = \
        split_train_test(X,y, test_size, n_labeled, random_state=1)

    trn_ds2 = copy.deepcopy(trn_ds)
    trn_ds3 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    quota = len(y_train) - n_labeled    # number of samples to query

    # Comparing UncertaintySampling strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    model = LogisticRegression()
    E_in_1, E_out_1, fc1 = run(trn_ds, tst_ds, lbr, model, qs, quota)

    qs2 = RandomSampling(trn_ds2)
    model = LogisticRegression()
    E_in_2, E_out_2,fc2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota)

    ## ALCE
    cost_matrix = 2000. * np.random.rand(n_classes, n_classes)
    qs3 = ALCE(trn_ds3, cost_matrix, SVR())
    E_in_3, E_out_3,fc3 = run(trn_ds3, tst_ds, lbr, model, qs3, quota)

     # error rate.
    query_num = np.arange(1, quota + 1)
    #plt.plot(query_num, E_in_1, 'b', label='qs Ein')
   # plt.plot(query_num, E_in_2, 'r', label='random Ein')
    # plt.plot(query_num, E_out_1, 'g', label='uncertain Eout')
    # plt.plot(query_num, E_out_2, 'k', label='random Eout')
    # plt.plot(query_num, E_out_3, 'b', label='quire Eout')


    plt.plot(query_num, fc1, 'g', label='uncertain F1')
    plt.plot(query_num, fc2, 'k', label='random F1')
    plt.plot(query_num, fc3, 'b', label='alce F1')

    
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()




