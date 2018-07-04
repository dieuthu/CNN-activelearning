import numpy as np
from os.path import join
import csv
import pandas as pd

datafolder = '../'

def load_zho_tfidplus_features():
    """
    Return X: a matrix of 275*103 and y an array of 275
    
    """
    filename = join(datafolder, 'features/Chinese/tfidf+feature.csv')
    df = pd.read_csv(filename)
    
    # df.values is an ndarray 
    X = df.values

    filename = join(datafolder, 'features/Chinese/label.csv')
    df = pd.read_csv(filename)
    y = df.values
    y=np.reshape(y, (636,))
    
    return X,y

def load_eng_all_features():
    """
    Return X: a matrix of 275*103 and y an array of 275
    
    """
    filename = join(datafolder, 'features/English/EnglishAllFeature102.csv')
    df = pd.read_csv(filename, header=None)
    
    # df.values is an ndarray 
    X = df.values

    filename = join(datafolder, 'features/English/label.csv')
    df = pd.read_csv(filename, header=None)
    y = df.values

    print (X.shape, y.shape)
    y=np.reshape(y, (275,))
    
    return X,y
    
if __name__=='__main__':
    #X, y = load_zho_tfidplus_features()
    #print (X.shape, y.shape)
    X, y = load_eng_all_features()
    print(X)
    print y
    
