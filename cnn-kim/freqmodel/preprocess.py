from sklearn.utils import shuffle
from os import listdir
from os.path import join
import re

datafolder = '../../data/'
senspliter = re.compile('[!?.\n]')

def merge_raw_data(folder):
    """
    merge all the text files into one big file 
    where one line for one document
    """
    # list folder
    files = listdir(folder)
    files = sorted(files)
    print (files)
    
    # read files
    outfile = join(folder, '..', folder + '.txt')
    with open(outfile, 'w') as fout:
        for filename in files:
            filepath = join(folder, filename)
            label = filename.split('_')[0]
            
            with open(filepath, 'r') as f:
                content = ''
                for line in f:
                    content += line
            content = content.replace('\n', ' ')
            fout.write(label + "\t" + content + '\n')


            
def read_and_split_data(filepath, random_state=0):
    """
    Return data where data["train_x"], data["train_y"]
                      data["dev_x"], data["dev_y"]
                      data["test_x"], data["test_y"]
    are training set, development set, and testing dataset
    """
    data = {}
    x, y = [], []
    with open(filepath) as f:
        for line in f:
            label = line.split('\t')[0].strip()
            content = line.split('\t')[1]

            y.append(label)
            x.append(line.split('\t')[1])

            # tks = senspliter.split(content)
            # for tk in tks:
            #     y.append(label)
            #     x.append(tk)
        x, y = shuffle(x, y, random_state=random_state)

    # split data into 70% for training, 30% for testing
    # among training data, take 10% for dev, the rest is for training
    train_size = len(y)*7//10
    test_size = len(y) - train_size
    dev_size = train_size//10

    data['test_x'] = x[train_size:]
    data['test_y'] = y[train_size:]
    data['dev_x'] = x[:dev_size]
    data['dev_y'] = y[:dev_size]
    data['train_x'] = x[dev_size:train_size]
    data['train_y'] = y[dev_size:train_size]

    return data
            
def read_and_split_data2(filepath, random_state=0, nfold=3, foldid=0):
    """
    Return data where data["train_x"], data["train_y"]
                      data["dev_x"], data["dev_y"]
                      data["test_x"], data["test_y"]
    are training set, development set, and testing dataset
    """
    data = {}
    x, y = [], []
    with open(filepath) as f:
        for line in f:
            label = line.split('\t')[0].strip()
            content = line.split('\t')[1]

            y.append(label)
            x.append(line.split('\t')[1])

            # tks = senspliter.split(content)
            # for tk in tks:
            #     y.append(label)
            #     x.append(tk)
        x, y = shuffle(x, y, random_state=random_state)


    nsamples =  len(y)
    nsample_per_fold = nsamples//nfold
    idx = range(len(y))
    foldid += 1
    
    if foldid != nfold:
        test_idx = list(range((foldid-1)*nsample_per_fold, foldid*nsample_per_fold))
    else: test_idx = list(range((foldid-1)*nsample_per_fold, nsamples))

    train_idx = [i for i in idx if i not in test_idx]
    train_size = len(train_idx)
    dev_size = train_size//10
    dev_idx = train_idx[:dev_size]
    train_idx = [i for i in train_idx if i not in dev_idx]
    
    data['test_x'] = [x[item] for item in test_idx]
    data['test_y'] = [y[item] for item in test_idx]
    data['dev_x'] = [x[item] for item in dev_idx]
    data['dev_y'] = [y[item] for item in dev_idx]
    data['train_x'] = [x[item] for item in train_idx]
    data['train_y'] = [y[item] for item in train_idx]

    return data
        
    
if __name__=="__main__":
#    read_and_split_data(join(datafolder, 'txt', 'English_dataset.txt'))
    data=read_and_split_data2(join('../../datalight', 'txt', 'English_dataset.txt'))
    total = 0
    for key, val in data.items():
        if 'x' in key:
            print (key, len(val))
            total += len(val)
    print (total)
    
    
                
    
    
