import os
from sklearn.model_selection import train_test_split
from datautil import *


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
    
    

def split_train_test_feature(X,y,l_train_label,l_test_label, n_class):    
    l_train = []
    l_test = []
    #level = ['1','2','3','4']
    d = {} #label: (#train, #test, #total)

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
            l_train.append(X[cur_ind])
            cur_ind+=1
        for t in range(0,d[si][1]):
            print('===== ' + str(cur_ind))
            l_test.append(X[cur_ind])
            cur_ind+=1

    print(len(l_train))
    print(len(l_test))
    return (l_train, l_test)
                
        
    
    
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
	
	
def write_to_files(fn,  dire, l, labels):
    f = open(fn, 'w')
    for i in range(0, len(l)):
        f2 = open(os.path.join(dire,l[i]),'r')
        content = f2.read().strip().replace('\n', ' ').replace('\r',' ')
        f2.close()
        f.write(labels[i] + ' ' + content + '\n')        
    f.close()    
    
    
dire = '../data/English_dataset'
#dire = '../data/Chinese_datasetutf_8'


l_train, l_train_label, l_test, l_test_label = split_train_test(dire)


#write_to_files('zho_test', dire, l_test,l_test_label)
#write_to_files('zho_train', dire, l_train,l_train_label)

#print(l_test)
#write_to_files('eng_test', dire, l_test,l_test_label)
#write_to_files('eng_train', dire, l_train,l_train_label)


X,y = load_eng_all_features()
print(len(X))
print(len(y))
d = {}
for i in y:
    if i not in d:
        d[i]=1
    else:
        d[i]+=1
print(d)

lf_train, lf_test = split_train_test_feature(X, y, l_train_label, l_test_label,4)
import pickle
pickle.dump(lf_train,open('lf_train','wb'))
pickle.dump(lf_test,open('lf_test','wb'))