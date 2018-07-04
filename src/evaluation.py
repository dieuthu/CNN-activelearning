import codecs
import os
from sklearn.metrics import f1_score,accuracy_score
from baseline import *

from sklearn.model_selection import train_test_split




def read_data(dire):
    X = [] #input
    Y_true = [] 
    all_files = os.listdir(dire)
    for fn in all_files:
        if '.txt' in fn:
            f = codecs.open(dire+fn,'r',encoding='utf-8')
            level = int(fn.strip().split('_')[0])
            content = f.read().strip()
            X.append(content)
            Y_true.append(level)
    return (X,Y_true)
            
        

def get_accuracy_score(y_true,y_pred):
    return accuracy_score(y_true, y_pred)              

def get_f1_score(y_true,y_pred):
    return f1_score(y_true, y_pred, average=None)  


def run_NDC(X,lang,sentenceLength):
    Y_pred = []
    ndcDetect = NDCLevel(lang)
    for x in X:
        x_words = []
        if lang=="eng":
            x_words = x.strip().split() #Split by space for english
        #For Chinese, get each letter
        if lang=='zho':            
            for i in x:
                if i.strip()!='':
                    x_words.append(i.strip())
                    
        level = ndcDetect.detect_level(x_words, {KEY_AVG_SENTENCE_LENGTH:sentenceLength})
        Y_pred.append(level)
    return Y_pred
    
def run_FNDC(X,lang,sentenceLength):
    Y_pred = []
    print("reading frequency dictionary..")    
    dictFile = '../store/eng-word-byfreq.txt'
    if lang=='zho':
        dictFile = '../store/zho-word-byfreq.txt'
    freqdict = read_freqdict(dictFile)    
    fndcDetect = FNDCLevel(lang,freqdict,None,9)
    for x in X:
        x_words = x.strip().split()
        level = fndcDetect.detect_level(x_words, {KEY_AVG_SENTENCE_LENGTH:sentenceLength})
        Y_pred.append(level)
    return Y_pred
    
    
def get_accuracy(X, y, l_train_label, l_test_label, n_class):
    l_train = []
    l_test = []
    #level = ['1','2','3','4']
    d = {} #label: (#train, #test, #total)
    ncorrect = 0
    nall = 0 
    
    for i in l_train_label:
        if i not in d:
            d[i] = [1,0,1]
        else:
            d[i][0]+=1
            d[i][2]+=1
            
    for i in l_test_label:
        if i not in d:
            d[i] = [0,1,1]
        else:
            d[i][1]+=1
            d[i][2]+=1
        
    print(d)
    cur_ind = 0
    for i in range(1,n_class+1): #each label
        si = str(i)
        print('============Label ' + str(i))
        #for j in range(0,d[i][2]):
        for k in range(0,d[si][0]):
            print(cur_ind)
            #l_train.append(X[cur_ind])
            #Skip
            cur_ind+=1
        for t in range(0,d[si][1]):
            print('===== ' + str(cur_ind))
            #if 
            #l_test.append(X[cur_ind])
            cur_ind+=1    
            nall+=1
        
    
    
def split_train_test(dire):
    d_train = {} #class 1: [fn], class 2: [fn]...
    d_test = {} #class 1: [fn], class 2: [fn]...
    d_all = {} #class 1: [fn]
    l_train = []
    l_train_label = []
    l_test = []
    l_test_label = []
    
    
    for fn in os.listdir(dire):
        if '.txt' in fn:
            label  = fn.strip().split('_')[0]
            if label not in d_all:
                d_all[label] = [fn]
            else: 
                d_all[label].append(fn)
    for l in d_all:
        d_train[l] = []
        d_test[l] = []
        n_items = len(d_all[l])
        n_train = int(n_items*0.7) + (n_items*7 % 100 > 0) #split 70% for training, round up
        n_test = n_items-n_train
        i = 0
        for item in d_all[l]:
            i+=1
            if i<n_train:
                d_train[l].append(item)
                l_train.append(item)
                l_train_label.append(l)
            else:
                d_test[l].append(item)
                l_test.append(item)
                l_test_label.append(l)                
    #return (d_train, d_test)
    return (l_train, l_train_label, l_test, l_test_label)
    
        
if __name__=='__main__':
    """TESTING ENGLISH"""
    # X, Y_true = read_data("../data/English_dataset/")
    # print(Y_true)
    # Y_predNDC = run_NDC(X,'eng',27)
    # Y_predFNDC = run_FNDC(X,'eng',27)
    # print(Y_predNDC)
    # print(get_f1_score(Y_true, Y_predNDC))
    # print(get_accuracy_score(Y_true, Y_predNDC))
    #
    # print(Y_predFNDC)
    # print(get_f1_score(Y_true, Y_predFNDC))
    # print(get_accuracy_score(Y_true, Y_predFNDC))
    
    
    #dire = '../data/English_dataset'
    #dire = '../data/Chinese_datasetutf_8'


    #l_train, l_train_label, l_test, l_test_label = split_train_test(dire)
    
    
    
    """TESTING CHINESE"""
    X, Y_true = read_data("../data/Chinese_datasetutf_8/")
    print(Y_true)
    Y_predNDC = run_NDC(X,'zho',100)
    Y_predFNDC = run_FNDC(X,'zho',100)
    print(Y_predNDC)
    print(get_f1_score(Y_true, Y_predNDC))
    print(get_accuracy_score(Y_true, Y_predNDC))    
    print(Y_predFNDC)
    print(get_f1_score(Y_true, Y_predFNDC))
    print(get_accuracy_score(Y_true, Y_predFNDC))