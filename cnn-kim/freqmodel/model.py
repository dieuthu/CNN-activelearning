import pdb
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy
import pickle
from os.path import join
from torch.nn.functional import softmax
from sklearn.metrics import f1_score
import re

def load_wordfreq_embeddings(word2vec_file, wordfreq_file, vocab=None):
    """
    Paramters:
    ---------
    wordfreq_file: header\n {data_line}
                   data_line:= <rank> \t <word> \t <freq_class> \t <freq_number>
    word2vec_file: gensim KeyedVectors
    vocab: a list of words

    Returns:
    ----
    wv_matrix
    wv_freq
    vocab
    """
    # get frequency embeding
    FRE_DIM = 1
    freq_level = {}
    topwords = []
    with open(wordfreq_file) as f:
        f.readline()
        for line in f:
            tks = line.strip().split("\t")
            freq_level[tks[1]] = [int(tks[2])]
            if len(topwords) < 10000: topwords.append(tks[1])
        f.close()
    if vocab == None: vocab = topwords
    
    wv_freq = []
    for word in vocab:
        if word in freq_level:
            wv_freq.append(freq_level[word])
        else: wv_freq.append([19])

    # one for unknown and zero pedding
    wv_freq.append([19])
    wv_freq.append(np.zeros(FRE_DIM).astype("float32"))
    wv_freq = np.array(wv_freq)

    # get word embeddings
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
    WORD_DIM = word_vectors.get_vector(word_vectors.index2word[0]).shape[0]
    #WORD_DIM = 300    
    wv_matrix = []
    for word in vocab:
       if word in word_vectors.vocab:
           l = word_vectors.get_vector(word)
           wv_matrix.append(l)
       else:
           wv_matrix.append(np.random.uniform(-0.01, 0.01, WORD_DIM).astype("float32"))

    wv_matrix.append(np.random.uniform(-0.01, 0.01, WORD_DIM).astype("float32"))
    wv_matrix.append(np.zeros(WORD_DIM).astype("float32"))
    wv_matrix = np.array(wv_matrix)


    return wv_matrix, wv_freq, vocab

class Preprocessor(object):
    """
    """
    def __init__(self, wordvec_file=None, freqlevel_file=None, topwords_as_vocab=False):
        """
        wordvec_file and freqlevel_file are needed only during training
        """
        self.wordvec_file = wordvec_file
        self.freqlevel_file = freqlevel_file
        self.topwords_as_vocab = topwords_as_vocab
        return

    def preprocess_doc(self, doc):
        """
        Parameters:
        ----
        doc: string

        Returns:
        ---
        a list of words/tokens
        """
        return doc.split()
        
    def preprocess_sent(self, doc):
        """
        Parameters:
        ----
        doc: string

        Returns:
        ---
        a list of sentences
        """
        return re.split('\.|\?|\!',doc)
        
        
    def prepare_train(self, doclist, classes, options= None):
        """
        doclist: a list of documents in the training dataset.
        each document is a string
        """
        assert(self.wordvec_file != None and self.freqlevel_file != None)
        if (options != None):
            params = options
        else: 
            params = {
                "MODEL": "rand",
                #    "MODEL_FILE": "model.pkl",
                "EARLY_STOPPING": False,
                "EPOCH": 100,
                "LEARNING_RATE": 0.1,
                "MAX_DOC_LEN": 100,
                "BATCH_SIZE": 10,
                "WORD_DIM": None,
                "VOCAB_SIZE": None,
                "CLASS_SIZE": None,
                "SENT_AVG": True,
                "FILTERS": [3, 4, 5],
                "FILTER_NUM": [100, 100, 100],
                "DROPOUT_PROB": 0.5,
                "NORM_LIMIT": 3,
                "MAX_SENT_LEN": 50,
                "GPU": -1 # the number of gpu to be used
            }        
        processed_docs = None    
        if params["SENT_AVG"]:
            processed_docs = []
            for doc in doclist:
                doc_tmp = []
                for sent in self.preprocess_sent(doc):
                    doc_tmp.append(self.preprocess_doc(sent))
                processed_docs.append(doc_tmp)
            print(len(processed_docs))
                
            max_sent_len = max([len(doc) for doc in processed_docs])
            if not self.topwords_as_vocab:
                vocab = sorted(list(set([w for doc in processed_docs for sent in doc for w in sent])))
            else: vocab = None

            wv_matrix, wv_freq, vocab = load_wordfreq_embeddings(self.wordvec_file, self.freqlevel_file, vocab)
            params["MAX_DOC_LEN"] = max_sent_len
            print("set max doc len", max_sent_len)
            params["WORD_DIM"] = wv_matrix[0].shape[0]
            params["VOCAB_SIZE"] = len(vocab)
            params["CLASS_SIZE"] = len(set(classes))
            
            return wv_matrix, wv_freq, vocab, params, processed_docs

        else:
            processed_docs = [self.preprocess_doc(doc) for doc in doclist]
            max_doc_len = max([len(doc) for doc in processed_docs])
            if not self.topwords_as_vocab:
                vocab = sorted(list(set([w for doc in processed_docs for w in doc])))
            else: vocab = None
        
            # load word and freq embeddings
            wv_matrix, wv_freq, vocab = load_wordfreq_embeddings(self.wordvec_file, self.freqlevel_file, vocab)

            params["MAX_DOC_LEN"] = max_doc_len
            params["WORD_DIM"] = wv_matrix[0].shape[0]
            params["VOCAB_SIZE"] = len(vocab)
            params["CLASS_SIZE"] = len(set(classes))

            return wv_matrix, wv_freq, vocab, params, processed_docs
        

class ReadlevelClassifier():
    """ Perform readability assessment with cnn multichannel sep
    
    """
    def __init__(self, preprocessor, useGPU=True):
        self.preprocessor = preprocessor
        self.model = None
        self.wv_matrix = None
        self.wv_freq = None
        self.vocab = None
        self.params = None
        self.classes = None
        self.word_to_idx = None
        self.useGPU = useGPU

    @staticmethod
    def save_model(classifier, modelfile):
        """
        Save cnn model
        """
        pickle.dump(classifier, open(modelfile, "wb"))

    @staticmethod
    def load_model(modelfile):
        """
        """
        model = pickle.load(open(modelfile, "rb"))
        return model

    def predict(self, docs, need_preprocess=True):
        """
        Parameters:
        ----
        docs: a list of docs
              where each doc is a string (need_preprocess=true) or
                             a list of word ids (need_preprocess=false)

        Returns:
        ---
        Class levels

        """
        output = self.predict_proba(docs, need_preprocess)
        labels = [self.classes[i] for i in np.argmax(output, axis=1)]
        return labels

    def cuda(self):
        self.useGPU = True
        if (self.model != None):
            self.model.cuda(self.params["GPU"])

    def cpu(self):
        self.useGPU = False
        if (self.model != None):
            self.model.cpu()

    def get_classes():
        return self.classes

    def __create_longtensor(self, arr):
        if self.useGPU:
            return Variable(torch.LongTensor(arr)).cuda(self.params["GPU"])
        else: return Variable(torch.LongTensor(arr))
            
    def __to_word_idx(self, docs):
        """
        each doc is a list of words (string)
        """
        returned = []
        max_len = self.params["MAX_DOC_LEN"]
        if (self.params["SENT_AVG"]):
            max_len = self.params["MAX_SENT_LEN"]


        for doc in docs:
            word2ids = list()
            for w in doc:
                if w in self.word_to_idx: word2ids.append(self.word_to_idx[w])
                if (len(word2ids) >= max_len):
                    break

            if (len(word2ids) < max_len):
                word2ids += [self.params["VOCAB_SIZE"] + 1]*(max_len-len(word2ids))
            returned.append(word2ids)
            
        return returned

    def __add_sentences(self, docs, max_sent):
        print("Input docs: ")
        out = []
        for doc in docs:
            out.append(self.__to_word_idx(doc))
        print("Doc length originally ", len(docs))
        if len(docs) < max_sent:
            for i in range(len(docs), max_sent):
                out.append([self.params["VOCAB_SIZE"]])
        print("Out length: ")
        print(len(out))
        return out


    def test(self, X, y):
        pred = self.predict(X, need_preprocess=False)
        acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)
        f1 = f1_score(y, pred, average='macro')
        return acc, f1

    def fit_sentavg(self, X, y, dev_X=[], dev_y=[], test_X=[], test_y=[]):
        """
        X: a list of documents, each is a list of words
        y: a list of corresponding classes

        """
        self.classes = sorted(list(set(y)))
        
        output =  self.preprocessor.prepare_train(
            X + dev_X + test_X, self.classes)

        self.wv_matrix, self.wv_freq, self.vocab, self.params, docs = output
        self.word_to_idx = {w:i for i,w in enumerate(self.vocab)}
        self.params["WV_MATRIX"] = self.wv_matrix
        self.params["FEA_VEC"] = self.wv_freq

        print("=" * 20 + "INFORMATION" + "=" * 20)
        print("MODEL:", self.params["MODEL"])
        print("VOCAB_SIZE:", self.params["VOCAB_SIZE"])
        print("CLASS_SIZE:", self.params["CLASS_SIZE"])
        print("EPOCH:", self.params["EPOCH"])
        print("LEARNING_RATE:", self.params["LEARNING_RATE"])
        print("MAX_DOC_LEN:", self.params["MAX_DOC_LEN"])
        print("EARLY_STOPPING:", self.params["EARLY_STOPPING"])
        print("=" * 20 + "INFORMATION" + "=" * 20)
        
        # convert X, dev_X, test_X into a list of doc, each is a list of word idxes
        X = docs[:len(X)]
        dev_X = docs[len(X):len(X)+len(dev_X)]
        test_X = docs[len(X)+len(dev_X):]

        X_sents = self.__add_sentences(X, self.params["MAX_DOC_LEN"])
        dev_X_sents = self.__add_sentences(dev_X, self.params["MAX_DOC_LEN"])        
        test_X_sents = self.__add_sentences(test_X, self.params["MAX_DOC_LEN"])
            

        if self.useGPU:
            self.model = CNN(**self.params).cuda(self.params["GPU"])
        else: self.model = CNN(**self.params)
            
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.Adadelta(parameters, self.params["LEARNING_RATE"])
        criterion = nn.CrossEntropyLoss()

        pre_dev_acc = 0
        max_dev_acc = 0
        max_test_acc = 0
        for e in range(self.params["EPOCH"]):
            X, y = shuffle(X_sents, y)
            for i in range(0, len(X), self.params["BATCH_SIZE"]):


                batch_x = X[i:i + batch_range]

                #print(batch_x)
                print(len(batch_x))
                print(len(batch_x[0]))
                print(len(batch_x[1]))
                batch_x = self.__create_longtensor(batch_x)
                batch_y = self.__create_longtensor(batch_y)
                
                optimizer.zero_grad()
                self.model.train()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)

                loss.backward()
                nn.utils.clip_grad_norm(parameters, max_norm=self.params["NORM_LIMIT"])
                optimizer.step()

            dev_acc, test_acc, dev_f1, test_f1 = 0,0,0,0
            if len(dev_X_sents) != 0 or len(test_X_sents)!=0:
                if len(dev_X_sents) != 0:
                    dev_acc, dev_f1 = self.test(dev_X_sents, dev_y)
                if len(test_X_sents) != 0:
                    test_acc, test_f1 = self.test(test_X_sents, test_y)
                print("epoch:", e + 1, "/ dev_acc:", dev_acc, "/ test_acc:", test_acc)
            
            if self.params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
                print("early stopping by dev_acc!")
                break
            else:
                pre_dev_acc = dev_acc
                
            if dev_acc >= max_dev_acc:
                max_dev_acc = dev_acc
                max_test_acc = test_acc
                best_model = copy.deepcopy(self.model)

        print("max dev acc:", max_dev_acc, "test acc:", max_test_acc)
        self.model = best_model
        return best_model        
        
    def fit(self, X, y, dev_X=[], dev_y=[], test_X=[], test_y=[], params=None):
        """
        X: a list of documents, each is a list of words
        y: a list of corresponding classes

        """
        self.classes = sorted(list(set(y)))
        output =  self.preprocessor.prepare_train(
            X + dev_X + test_X, self.classes, options=params)

        self.wv_matrix, self.wv_freq, self.vocab, self.params, docs = output
        self.word_to_idx = {w:i for i,w in enumerate(self.vocab)}
        self.params["WV_MATRIX"] = self.wv_matrix
        self.params["FEA_VEC"] = self.wv_freq

        print("=" * 20 + "INFORMATION" + "=" * 20)
        print("MODEL:", self.params["MODEL"])
        print("VOCAB_SIZE:", self.params["VOCAB_SIZE"])
        print("CLASS_SIZE:", self.params["CLASS_SIZE"])
        print("EPOCH:", self.params["EPOCH"])
        print("LEARNING_RATE:", self.params["LEARNING_RATE"])
        print("MAX_DOC_LEN:", self.params["MAX_DOC_LEN"])
        print("EARLY_STOPPING:", self.params["EARLY_STOPPING"])
        print("=" * 20 + "INFORMATION" + "=" * 20)
        
        # convert X, dev_X, test_X into a list of doc, each is a list of word idxes
        X = docs[:len(X)]
        dev_X = docs[len(X):len(X)+len(dev_X)]
        test_X = docs[len(X)+len(dev_X):]

        X = self.__to_word_idx(X)
        dev_X = self.__to_word_idx(dev_X)
        test_X = self.__to_word_idx(test_X)

        if self.useGPU:
            self.model = CNN(**self.params).cuda(self.params["GPU"])
        else: self.model = CNN(**self.params)
            
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = optim.Adam(parameters, self.params["LEARNING_RATE"])
        criterion = nn.CrossEntropyLoss()

        pre_dev_acc = 0
        max_dev_acc = 0
        max_test_acc = 0
        for e in range(self.params["EPOCH"]):
            X, y = shuffle(X, y)
            for i in range(0, len(X), self.params["BATCH_SIZE"]):
                batch_range = min(self.params["BATCH_SIZE"], len(X) - i)

                batch_x = X[i:i + batch_range]
                batch_y = [self.classes.index(c) for c in y[i:i + batch_range]]

                batch_x = self.__create_longtensor(batch_x)
                batch_y = self.__create_longtensor(batch_y)
                
                optimizer.zero_grad()
                self.model.train()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)

                loss.backward()
                nn.utils.clip_grad_norm(parameters, max_norm=self.params["NORM_LIMIT"])
                optimizer.step()

            dev_acc, test_acc, dev_f1, test_f1 = 0,0,0,0
            if len(dev_X) != 0 or len(test_X)!=0:
                if len(dev_X) != 0:
                    dev_acc, dev_f1 = self.test(dev_X, dev_y)
                if len(test_X) != 0:
                    test_acc, test_f1 = self.test(test_X, test_y)
#                print("epoch:", e + 1, "/ dev_acc:", dev_acc, "/ test_acc:", test_acc)
            
            if self.params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
                print("early stopping by dev_acc!")
                break
            else:
                pre_dev_acc = dev_acc
                
            if dev_acc >= max_dev_acc:
                max_dev_acc = dev_acc
                max_test_acc = test_acc
                best_model = copy.deepcopy(self.model)

        print("max dev acc:", max_dev_acc, "test acc:", max_test_acc)
        self.model = best_model
        return best_model

    def predict_proba(self, docs, need_preprocess=True):
        """
        Parameters:
        ----
        docs: a list of docs
              where each doc is a string (need_preprocess=true) or
                             a list of word ids (need_preprocess=false)

        Returns:
        ---
        Probabilities of level classes

        """
        if (self.model == None):
            raise Exception('Model needs to be trained or loaded from a pretrained modelfile.')

        self.model.eval()
        if need_preprocess:
            X = [self.preprocessor.preprocess_doc(doc) for doc in docs]
            X = self.__to_word_idx(X)
        else: X = docs
        
        X = self.__create_longtensor(X)
        if (self.useGPU):
            output = self.model(X).cpu()
        else:  output = self.model(X)
        m = nn.Softmax(dim=1)
        output = m(output).data.numpy()
        return output
    

class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()
        
        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_DOC_LEN = kwargs["MAX_DOC_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1
        self.FRE_DIM = 1
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.SENT_AVG = kwargs["SENT_AVG"]
        #self.SENT_AVG = False

        self.choice = 'wfc'

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if self.MODEL in ["static", "non-static", "rand", "multichannel-sep", "multichannel-com"]:
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False
            elif self.MODEL in ["multichannel-sep", "multichannel-com"]:
                self.FEA_VEC = kwargs["FEA_VEC"]                
                self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.FRE_DIM, padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.FEA_VEC))
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 1

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, 'conv_{0}'.format(i), conv)

        nFC = sum(self.FILTER_NUM)
        MAX_LENGTH = self.MAX_DOC_LEN
        if (self.SENT_AVG):
            MAX_LENGTH = self.MAX_SENT_LEN

        if self.MODEL in ["multichannel-sep", "multichannel-com"]:
            if self.choice=='wfc':
                nFC+= MAX_LENGTH*self.FRE_DIM
            else:
                nFC = MAX_LENGTH*self.FRE_DIM
        print("nFC = " + str(nFC))
        self.fc = nn.Linear(nFC, self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, 'conv_{0}'.format(i))

    def forward(self, inp):
        if (self.SENT_AVG):
            x_docs = 0
            for doc in inp: #for each doc
                for sent in doc: #each sentence in a document
                    x_doc = []

                    x = self.embedding(sent).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)

                    if self.MODEL == "multichannel-com": # first concatenate then conv
                        x2 = self.embedding2(sent).view(-1, 1, 3 * self.MAX_SENT_LEN)
                        x = torch.cat((x, x2), 2)

                    conv_results = [
                        F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                            .view(-1, self.FILTER_NUM[i])
                        for i in range(len(self.FILTERS))]
                    x = torch.cat(conv_results, 1)

                    if self.MODEL == "multichannel-sep":        
                        x2 = self.embedding2(sent).view(-1, 1, self.FRE_DIM * self.MAX_SENT_LEN)
                        x2 = x2.view(x2.shape[0],-1)

                        if self.choice=='wfc':
                            x = torch.cat((x, x2),1)
                        elif self.choice=='fc':
                            x = x2
                    x_doc.append(x)
                x_av = 0
                for s in x_doc:
                    x_av+=doc
                x_av=x_av/len(x_doc)
                x_docs = torch.stack((x_docs,x_av),0)
            x_docs = F.dropout(x_docs, p=self.DROPOUT_PROB, training=self.training)
            x_docs = self.fc(x_docs)
            return x_docs
        else:            
            x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_DOC_LEN)
            if self.MODEL == "multichannel-com": # first concatenate then conv
                x2 = self.embedding2(inp).view(-1, 1, 3 * self.MAX_DOC_LEN)
                x = torch.cat((x, x2), 2)

            conv_results = [
                F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_DOC_LEN - self.FILTERS[i] + 1)
                    .view(-1, self.FILTER_NUM[i])
                for i in range(len(self.FILTERS))]
            x = torch.cat(conv_results, 1)

            if self.MODEL == "multichannel-sep":        
                x2 = self.embedding2(inp).view(-1, 1, self.FRE_DIM * self.MAX_DOC_LEN)
                x2 = x2.view(x2.shape[0],-1)

                if self.choice=='wfc':
                    x = torch.cat((x, x2),1)
                elif self.choice=='fc':
                    x = x2


            x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
            x = self.fc(x)
            return x


