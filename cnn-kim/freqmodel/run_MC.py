import pdb
from model import CNN
import utils

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy
import pickle
from os.path import join
from torch.nn.functional import softmax
from preprocess import read_and_split_data

datafolder = '../../data'
dataset = 'eng'
#WORD_DIM = 300 #English
WORD_DIM = 200 #Chinese
FRE_DIM = 1 #Number of dimension in frequency embedding
word_freq_file = join(datafolder, 'freqdict', 'english-word-byfreq.txt')
#word2vec_file = 'GoogleNews-vectors-negative300.bin'
word2vec_file = join(datafolder, 'word2vec', 'eng.model.bin')

if dataset == 'zho':
    WORD_DIM = 200
    word_freq_file = join(datafolder, 'freqdict', 'zho-word-byfreq.txt')
    word2vec_file = join(datafolder, 'word2vec', 'zho.model2.bin')


    
def train(data, params):
    if params["MODEL"] in ["multichannel-sep","multichannel-com"]:
        print("Loading frequency level of words..")
        #f = open('english-word-byfreq.txt','r')
        f = open(word_freq_file,'r')
        freq_level = {} #word: freqLevel
        f.readline() #ignore header
        for line in f:
            lines = line.strip().split('\t')#
            freq_level[lines[1]] = [int(lines[2])]
            #freq_level[lines[1]] = [int(lines[0]), int(lines[2]), int(lines[3])]
        f.close()    
            
        wv_freq = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in freq_level:
                wv_freq.append(freq_level[word])
            else:
                #wv_freq.append([30000,19,1000]) #not in the list, consider difficult words
                wv_freq.append([19])

        # one for UNK and one for zero padding
        #wv_freq.append([30000,19,1000])
        wv_freq.append([19])
        #wv_freq.append(np.zeros(3).astype("float32"))
        wv_freq.append(np.zeros(FRE_DIM).astype("float32"))
        wv_freq = np.array(wv_freq)
        params["FEA_VEC"] = wv_freq
        
        
    #if params["MODEL"] != "rand" and params["MODEL"] != "multichannel":
    if params["MODEL"] != "rand":
        # load word2vec
        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)

        #word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                l = word_vectors.word_vec(word)
                wv_matrix.append(l)
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, WORD_DIM).astype("float32"))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, WORD_DIM).astype("float32"))
        wv_matrix.append(np.zeros(WORD_DIM).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix
        print(wv_matrix.size)        

    return do_train(data, params)
#     model = CNN(**params).cuda(params["GPU"])

#     parameters = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
#     criterion = nn.CrossEntropyLoss()

#     pre_dev_acc = 0
#     max_dev_acc = 0
#     max_test_acc = 0
#     for e in range(params["EPOCH"]):
# #        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

#         for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
#             batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

#             batch_x = [[data["word_to_idx"][w] for w in sent] +
#                        [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
#                        for sent in data["train_x"][i:i + batch_range]]

#             batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]
#             print (batch_x[0][0:10], batch_y[0])
#             batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
#             batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])

#             optimizer.zero_grad()
#             model.train()
#             pred = model(batch_x)
#             loss = criterion(pred, batch_y)
#             print (loss)
#             loss.backward()
#             nn.utils.clip_grad_norm(parameters, max_norm=params["NORM_LIMIT"])
#             optimizer.step()

#         dev_acc = test(data, model, params, mode="dev")
#         test_acc = test(data, model, params)
#         print("epoch:", e + 1, "/ dev_acc:", dev_acc, "/ test_acc:", test_acc)

#         if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
#             print("early stopping by dev_acc!")
#             break
#         else:
#             pre_dev_acc = dev_acc

#         if dev_acc > max_dev_acc:
#             max_dev_acc = dev_acc
#             max_test_acc = test_acc
#             best_model = copy.deepcopy(model)

#     print("max dev acc:", max_dev_acc, "test acc:", max_test_acc)
#     return best_model


def test(data, model, params, mode="test"):
    model.eval()

    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]

    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
    y = [data["classes"].index(c) for c in y]

    output = model(x).cpu().data.numpy() # put through softmax
    pred = np.argmax(output, axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc


def do_train(data, params):
    model = CNN(**params).cuda(params["GPU"])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    pre_dev_acc = 0
    max_dev_acc = 0
    max_test_acc = 0
    for e in range(params["EPOCH"]):
#        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]

            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]
            print (batch_x[0][0:10], batch_y[0])
            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            print (loss)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        dev_acc = test(data, model, params, mode="dev")
        test_acc = test(data, model, params)
        print("epoch:", e + 1, "/ dev_acc:", dev_acc, "/ test_acc:", test_acc)

        if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_dev_acc = dev_acc

        if dev_acc > max_dev_acc:
            max_dev_acc = dev_acc
            max_test_acc = test_acc
            best_model = copy.deepcopy(model)

    print("max dev acc:", max_dev_acc, "test acc:", max_test_acc)
    return best_model


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="rand", help="available models: rand, static, non-static, multichannel-com, multichannel-sep")
    parser.add_argument("--dataset", default="eng", help="available datasets: eng, zho")
    parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="learning rate")
    parser.add_argument("--gpu", default=-1, type=int, help="the number of gpu to be used")

    options = parser.parse_args()
    data = utils.read_TREC(options.dataset)
    #data = read_and_split_data(join(datafolder,  'txt', 'English_dataset.txt'))

    for i in range(len(data['train_x'])):
        data['train_x'][i] =  data['train_x'][i].split()

    for i in range(len(data['dev_x'])):
        data['dev_x'][i] =  data['dev_x'][i].split()

    for i in range(len(data['test_x'])):
        data['test_x'][i] =  data['test_x'][i].split()

    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))

    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 50,
        "WORD_DIM": WORD_DIM,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": options.gpu
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("MAX_SENT_LEN:", params["MAX_SENT_LEN"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params)
        if params["SAVE_MODEL"]:
            utils.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        model = utils.load_model(params).cuda(params["GPU"])

        test_acc = test(data, model, params)
        print("test acc:", test_acc)


if __name__ == "__main__":
    main()

