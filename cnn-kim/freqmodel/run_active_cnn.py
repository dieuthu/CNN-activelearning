import numpy as np
import pdb
import model as rlc
import copy
from os.path import join
from preprocess import read_and_split_data2
from libact.models.sklearn_adapter import SklearnProbaAdapter
from libact.base.interfaces import ProbabilisticModel
from libact.query_strategies import RandomSampling, UncertaintySampling
from variance_reduction import MyVarianceReduction
from uncertainty_sampling import MyUncertaintySampling
from random_sampling import MyRandomSampling
from libact.labelers import IdealLabeler
from libact.base.dataset import Dataset
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import pandas as pd
from datautil import new_result_item, get_set_random_states

class CNNWrapperClassifier(ProbabilisticModel):
    def __init__(self, db, cls, dev):
        self.docs, self.y = db
        self.cls = cls
        self.dev_X, self.dev_y = dev
        
    def train(self, dataset, *args, **kwargs):
        X_id, y = dataset.format_sklearn()
        print ('===='*20)

        print ('train with size', X_id.shape)
        X = self.__convert(X_id)

        params = {
            "MODEL": "multichannel-sep",
            #    "MODEL_FILE": "model.pkl",
            "EARLY_STOPPING": False,
            "EPOCH": 100,
            "LEARNING_RATE": 0.001,
            "MAX_DOC_LEN": 100,
            "BATCH_SIZE": 10,
            "WORD_DIM": None,
            "VOCAB_SIZE": None,
            "CLASS_SIZE": None,
            "SENT_AVG": False,
            "FILTERS": [3, 4, 5],
            "FILTER_NUM": [100, 100, 100],
            "DROPOUT_PROB": 0.5,
            "NORM_LIMIT": 3,
            "MAX_SENT_LEN": 50,
            "GPU": -1 # the number of gpu to be used
        }        

        self.cls.fit(X, y, self.dev_X, self.dev_y, params=params)

    def predict(self, feature, *args, **kwargs):
        X = self.__convert(feature)
        return self.cls.predict(X)

    def __convert(self, X_id):
        X = []
        for id_ in X_id:
            X.append(self.docs[id_[0]])
        return X
        
    def score(self, testing_dataset, *args, **kwargs):
        X_id, y = testing_dataset.format_sklearn()
        X = self.__convert(X_id)

        pred = self.cls.predict(X)
        acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)
        return acc

    
    def predict_real(self, feature, *args, **kwargs):
        # feature is just the id in the database
        X = self.__convert(feature)
        return self.cls.predict_proba(X)
    
    def predict_proba(self, feature, *args, **kwargs):
        X  = self.__convert(feature)
        return self.cls.predict_proba(X)

def run(trn_ds, tst_ds, lbr, model, qs, quota):
    result = []
    while quota > 0:
        # Standard usage of libact objects

        ask_ids = qs.make_query()
        X, y = zip(*trn_ds.data)
        for ask_id in ask_ids:
            lb = lbr.label(X[ask_id])
            trn_ds.update(ask_id, lb)
        quota = quota - len(ask_ids)

        model.train(trn_ds)
        X_tst, y_tst = tst_ds.format_sklearn()
        y_pred=model.predict(X_tst)
        fc=f1_score(y_tst, y_pred, average='macro')

        num_instances = trn_ds.len_labeled()
        print ('run', num_instances, ':', fc)
        result.append((num_instances,fc))
        
    return result

def run_analysis(trn_ds, tst_ds, lbr, model, qs, quota):
    result = []
    model.train(trn_ds)
    while quota > 0:
        # Standard usage of libact objects

        ask_ids = qs.make_query()
        X, y = zip(*trn_ds.data)
        ask_ids_tmp = []
        for ask_id in ask_ids:
            lb = lbr.label(X[ask_id])
            print (qs, 'adding id, label', ask_id, lb)
            ask_ids_tmp.append('{0}:{1}'.format(ask_id,lb))
            trn_ds.update(ask_id, lb)
        quota = quota - len(ask_ids)

        model.train(trn_ds)
        X_tst, y_tst = tst_ds.format_sklearn()
        y_pred=model.predict(X_tst)
        fc=f1_score(y_tst, y_pred, average='macro')
        
        num_instances = trn_ds.len_labeled()
        print ('run', num_instances, ':', fc)
        result.append((num_instances,fc, ask_ids_tmp))
        
    return result


def prepare_data(X_train, y_train, X_test, y_test,  n_labeled, random_state=0):
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    
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

    return trn_ds, tst_ds, fully_labeled_trn_ds, n_labeled


import itertools
# testing
def perfom_test(dataset_file, word2vec_file, word_freq_file, output_file):
    nfolds = 3
    name = ['experimen_no', 'algorithm', 'query_strategy', 'number_of_instances', 'f1']
    with open(output_file, 'w') as f:
        f.write('{0}\n'.format(name))
            
    for i, foldid in itertools.product(get_set_random_states(), range(nfolds)):
        result = []
        data = read_and_split_data2(dataset_file, random_state=i, nfold=nfolds, foldid=foldid)
        
        # convert to libact dataset
        docs = data['train_x'] + data['dev_x'] + data['test_x']
        trn_len = len(data['train_x'])
        dev_len = len(data['dev_x'])
        test_len = len(data['test_x'])

        X = np.asarray([[i] for i in range(len(docs))])
        X_train = np.asarray([X[i,:] for i in range(trn_len)])
        X_dev = np.asarray([X[i,:] for i in range(trn_len,trn_len+dev_len)])
        X_test = np.asarray([X[i,:] for i in range(trn_len+dev_len, len(docs))])

        y = data['train_y'] + data['dev_y'] + data['test_y']
        y_train = data['train_y']
        y_dev = data['dev_y']
        y_test = data['test_y']        
        

        n_labeled = 50
        
        tmp = prepare_data(X_train,y_train, X_test,y_test, n_labeled, random_state=3)
        trn_ds, tst_ds, fully_labeled_trn_ds, n_labeled = tmp
        trn_ds2 = copy.deepcopy(trn_ds)
        trn_ds3 = copy.deepcopy(trn_ds)
        lbr = IdealLabeler(fully_labeled_trn_ds)
        quota = len(y_train) - n_labeled    # number of samples to query
        batchsize = 10

        
        # Comparing UncertaintySampling strategy with RandomSampling.
        # model is the base learner, e.g. LogisticRegression, SVM ... etc.
        # train CNN Readability Classifier
        # train CNN Readability Classifier
        preprocessor = rlc.Preprocessor(word2vec_file, word_freq_file, topwords_as_vocab=True)
        cls = rlc.ReadlevelClassifier(preprocessor, useGPU=True)
        cls.cuda()
    

        wrapper = CNNWrapperClassifier((docs,y), cls, (data['dev_x'], data['dev_y']))
        qs = MyUncertaintySampling(trn_ds, method='lc', model=wrapper, batch_size=batchsize)
        lc_result = run(trn_ds, tst_ds, lbr, wrapper, qs, quota)
        for result_item in lc_result:
            num_instance, f1 = result_item
            item = new_result_item('state={0}/foldid={1}'.format(i,foldid), 'cnn-sep', 'lc', num_instance, f1)
            result.append(item)


        # random
        preprocessor = rlc.Preprocessor(word2vec_file, word_freq_file, topwords_as_vocab=True)
        cls = rlc.ReadlevelClassifier(preprocessor, useGPU=True)
        cls.cuda()
        
        wrapper = CNNWrapperClassifier((docs,y), cls, (data['dev_x'], data['dev_y']))
        qs2 = MyRandomSampling(trn_ds2, batch_size=batchsize)
        rs_result = run(trn_ds2, tst_ds, lbr, wrapper, qs2, quota)
        for result_item in rs_result:
            num_instance, f1 = result_item
            item = new_result_item('state={0}/foldid={1}'.format(i,foldid), 'cnn-sep', 'rs', num_instance, f1)
            result.append(item)

        #wrapper = CNNWrapperClassifier((docs,y), cls)
        #qs3 = MyVarianceReduction(trn_ds3, model=wrapper) doesn't work
        #vr_result = run(trn_ds3, tst_ds, lbr, wrapper, qs3, quota)
        #for result_item in vr_result:
        #    num_instance, f1 = result_item
        #    item = new_result_item(i, 'cnn-sep', 'vr', num_instance, f1)
        #    result.append(item)


        result = pd.DataFrame(columns=name, data=result)
        with open(output_file, 'a') as f:
            result.to_csv(f,header=False)


# testing
def perfom_analysis(dataset_file, word2vec_file, word_freq_file, output_file):
    nfolds = 3
    name = ['experimen_no', 'algorithm', 'query_strategy', 'number_of_instances', 'f1']
    with open(output_file, 'w') as f:
        f.write('{0}\n'.format(name))
            
    #for i, foldid in zip([0],[0]):#itertools.product(get_set_random_states(), range(nfolds)):
    for i, foldid in itertools.product(get_set_random_states(), range(nfolds)):
        result = []
        data = read_and_split_data2(dataset_file, random_state=i, nfold=nfolds, foldid=foldid)

        # convert to libact dataset
        docs = data['train_x'] + data['dev_x'] + data['test_x']
        trn_len = len(data['train_x'])
        dev_len = len(data['dev_x'])
        test_len = len(data['test_x'])

        X = np.asarray([[i] for i in range(len(docs))])
        X_train = np.asarray([X[i,:] for i in range(trn_len)])
        X_dev = np.asarray([X[i,:] for i in range(trn_len,trn_len+dev_len)])
        X_test = np.asarray([X[i,:] for i in range(trn_len+dev_len, len(docs))])

        y = data['train_y'] + data['dev_y'] + data['test_y']
        y_train = data['train_y']
        y_dev = data['dev_y']
        y_test = data['test_y']        
        
        n_labeled = 50
        
        tmp = prepare_data(X_train,y_train, X_test,y_test, n_labeled, random_state=3)
        trn_ds, tst_ds, fully_labeled_trn_ds, n_labeled = tmp
        trn_ds2 = copy.deepcopy(trn_ds)
        lbr = IdealLabeler(fully_labeled_trn_ds)
        quota = len(y_train) - n_labeled    # number of samples to query
#        quota = 10
        batchsize = 10

        
        # Comparing UncertaintySampling strategy with RandomSampling.
        # model is the base learner, e.g. LogisticRegression, SVM ... etc.
        # train CNN Readability Classifier
        # train CNN Readability Classifier
        print ('>>>>>>>>>>>Least Certainty ....')
        preprocessor = rlc.Preprocessor(word2vec_file, word_freq_file, topwords_as_vocab=True)
        cls = rlc.ReadlevelClassifier(preprocessor, useGPU=True)
        cls.cuda()
    

        wrapper = CNNWrapperClassifier((docs,y), cls, (data['dev_x'], data['dev_y']))
        qs = MyUncertaintySampling(trn_ds, method='entropy', model=wrapper, batch_size=batchsize)
        lc_result = run_analysis(trn_ds, tst_ds, lbr, wrapper, qs, quota)
        for result_item in lc_result:
            num_instance, f1, ask_ids = result_item
            item = new_result_item('state={0}/foldid={1}/ask_ids={2}'.format(i,foldid, ask_ids), 'cnn-sep', 'lc', num_instance, f1)
            result.append(item)


        # random
        print ('--------->Random Sampling ....')
        preprocessor = rlc.Preprocessor(word2vec_file, word_freq_file, topwords_as_vocab=True)
        cls = rlc.ReadlevelClassifier(preprocessor, useGPU=True)
        cls.cuda()
        
        wrapper = CNNWrapperClassifier((docs,y), cls, (data['dev_x'], data['dev_y']))
        qs2 = MyRandomSampling(trn_ds2, batch_size=batchsize)
        rs_result = run_analysis(trn_ds2, tst_ds, lbr, wrapper, qs2, quota)
        for result_item in rs_result:
            num_instance, f1, ask_ids = result_item
            item = new_result_item('state={0}/foldid={1}/ask_ids={2}'.format(i,foldid,ask_ids), 'cnn-sep', 'rs', num_instance, f1)
            result.append(item)

        #wrapper = CNNWrapperClassifier((docs,y), cls)
        #qs3 = MyVarianceReduction(trn_ds3, model=wrapper) doesn't work
        #vr_result = run(trn_ds3, tst_ds, lbr, wrapper, qs3, quota)
        #for result_item in vr_result:
        #    num_instance, f1 = result_item
        #    item = new_result_item(i, 'cnn-sep', 'vr', num_instance, f1)
        #    result.append(item)


        result = pd.DataFrame(columns=name, data=result)
        with open(output_file, 'a') as f:
            result.to_csv(f,header=False)


# testing
def visualize(dataset_file, word2vec_file, word_freq_file, output_file):
    nfolds = 3
    for i, foldid in zip([0],[0]):#itertools.product(get_set_random_states(), range(nfolds)):
    #for i, foldid in itertools.product(get_set_random_states(), range(nfolds)):
        result = []
        data = read_and_split_data2(dataset_file, random_state=i, nfold=nfolds, foldid=foldid)

        # convert to libact dataset
        docs = data['train_x'] + data['dev_x'] + data['test_x']
        trn_len = len(data['train_x'])
        dev_len = len(data['dev_x'])
        test_len = len(data['test_x'])

        X = np.asarray([[i] for i in range(len(docs))])
        X_train = np.asarray([X[i,:] for i in range(trn_len)])
        X_dev = np.asarray([X[i,:] for i in range(trn_len,trn_len+dev_len)])
        X_test = np.asarray([X[i,:] for i in range(trn_len+dev_len, len(docs))])

        y = data['train_y'] + data['dev_y'] + data['test_y']
        y_train = data['train_y']
        y_dev = data['dev_y']
        y_test = data['test_y']        
        
        n_labeled = 50
        
        tmp = prepare_data(X_train,y_train, X_test,y_test, n_labeled, random_state=3)
        trn_ds, tst_ds, fully_labeled_trn_ds, n_labeled = tmp
        lbr = IdealLabeler(fully_labeled_trn_ds)
#        quota = len(y_train) - n_labeled    # number of samples to query
        quota = 10
        batchsize = 10

        
        # Comparing UncertaintySampling strategy with RandomSampling.
        # model is the base learner, e.g. LogisticRegression, SVM ... etc.
        # train CNN Readability Classifier
        # train CNN Readability Classifier
        print ('>>>>>>>>>>>UnCertainty ....')
        preprocessor = rlc.Preprocessor(word2vec_file, word_freq_file, topwords_as_vocab=True)
        cls = rlc.ReadlevelClassifier(preprocessor, useGPU=True)
        cls.cuda()
    

        wrapper = CNNWrapperClassifier((docs,y), cls, (data['dev_x'], data['dev_y']))
        wrapper.train(trn_ds)
        import ipdb
#        ipdb.set_trace()
        
        unlabeled_entry_ids, X_pool = zip(*trn_ds.get_unlabeled_entries())
        dvalue = wrapper.predict_proba(X_pool)
        from sklearn.manifold import TSNE
        X_embedded = TSNE(n_components=2).fit_transform(dvalue)
        labels = np.asarray([[lbr.label(X_pool[i]) for i in range(len(unlabeled_entry_ids))]]).transpose()
        print (labels.shape, X_embedded.shape)
        X_embedded = np.concatenate((X_embedded, labels), axis=1)

        result = pd.DataFrame(data=X_embedded)
        with open(output_file, 'w') as f:
            result.to_csv(f,header=False)

        # get the output
        # visualize the output
        # get the most uncertainty ids and mark


def visualize_eng():
    datafolder = '../../data/'
    datalight_folder = '../../datalight/'
    dataset_file = join(datalight_folder, 'txt', 'English_dataset.txt')
    word_freq_file = join(datafolder, 'freqdict', 'english-word-byfreq.txt')
    #word2vec_file = 'GoogleNews-vectors-negative300.bin'
    word2vec_file = join(datafolder, 'word2vec', 'eng.model.bin')
    output_file = join(datalight_folder, 'result', 'eng-cnn-sep-48trn_unlabeledoutput_may10.csv')
    visualize(dataset_file, word2vec_file, word_freq_file, output_file)

def visualize_zho():
    datafolder = '../../data/'
    datalight_folder = '../../datalight/'
    dataset_file = join(datalight_folder, 'txt', 'Chinese_datasetutf_8.txt')
    word_freq_file = join(datafolder, 'freqdict', 'zho-word-byfreq.txt')
    #word2vec_file = 'GoogleNews-vectors-negative300.bin'
    word2vec_file = join(datafolder, 'word2vec', 'zho.model2.bin')
    output_file = join(datalight_folder, 'result', 'zho-cnn-sep-48trn_unlabeledoutput_may16.csv')
    visualize(dataset_file, word2vec_file, word_freq_file, output_file)

def test_eng():
    datafolder = '../../data/'
    datalight_folder = '../../datalight/'
    dataset_file = join(datalight_folder, 'txt', 'English_dataset.txt')
    word_freq_file = join(datafolder, 'freqdict', 'english-word-byfreq.txt')
    #word2vec_file = 'GoogleNews-vectors-negative300.bin'
    word2vec_file = join(datafolder, 'word2vec', 'eng.model.bin')
    output_file = join(datalight_folder, 'result', 'eng-cnn-sep-active.csv')
    perfom_analysis(dataset_file, word2vec_file, word_freq_file, output_file)

def test_zho():
    datafolder = '../../data/'
    datalight_folder = '../../datalight/'
    dataset_file = join(datalight_folder, 'txt', 'Chinese_datasetutf_8.txt')
    word_freq_file = join(datafolder, 'freqdict', 'zho-word-byfreq.txt')
    #word2vec_file = 'GoogleNews-vectors-negative300.bin'
    word2vec_file = join(datafolder, 'word2vec', 'zho.model2.bin')
    output_file = join(datalight_folder, 'result', 'zho-cnn-sep-active.csv')
    perfom_test(dataset_file, word2vec_file, word_freq_file, output_file)

if __name__=='__main__':
#    test_eng()
#    visualize_eng()
    visualize_zho()
    
#     cls.fit(data['train_x'], data['train_y'],
#             data['dev_x'], data['dev_y'],
#             data['test_x'], data['test_y'])
#     rlc.ReadlevelClassifier.save_model(cls, 'model.pkl')
# else: cls = rlc.ReadlevelClassifier.load_model('model.pkl')

# print ('model loaded/trained')
# cls.cpu()
# print (cls.classes)
# print (cls.predict(["I am a student !", ""]))
# print (cls.predict(["This is a  an . !"]))
# print (cls.predict(["This is a  kajfkdjakfdja  kdajkfjk very difficult hospital virus !"]))

