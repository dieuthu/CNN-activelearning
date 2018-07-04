# -*- coding:utf-8 -*-
from __future__ import division
import string
import numpy as np
import codecs
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy import sparse
import math

#read files
data=[]
target=[]
path1="../data/English_dataset/1_1-"
path2="../data/English_dataset/2_2-"
path3="../data/English_dataset/3_3-"
path4="../data/English_dataset/4_4-"

for i in range(1,73):
    f = codecs.open(path1+str(i)+".txt","r")
    data.append(f.read())
    target.append(1)

for i in range(1,97):
    f = codecs.open(path2+str(i)+".txt","r")
    data.append(f.read())
    target.append(2)

for i in range(1,61):
    f = codecs.open(path3+str(i)+".txt","r")
    data.append(f.read())
    target.append(3)

for i in range(1,49):
    f = codecs.open(path4+str(i)+".txt","r")
    data.append(f.read())
    target.append(4)

x=np.array(data)
y=np.array(target)
# print x.shape
# print y.shape

#simple cross_validation
train_data, test_data, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# print train_data.shape
# print test_data.shape

#train
#feature1:tf-idf
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data)
tf_transformer = TfidfTransformer(use_idf=True)
X_train = tf_transformer.fit_transform(X_train_counts)

#feature2:number of sentences per text
sentences=[]
for i in range(0,len(train_data)):
     sentence=train_data[i].count("\n")
     sentences.append(sentence)
sentences=np.array(sentences)
sentences.shape=(len(train_data),1)
X_train=sparse.hstack((X_train,sentences))

#feature3:average and maximum number of tokens per sentence
average=[]
max=[]
for i in range(0,len(train_data)):
     token=nltk.word_tokenize(train_data[i])
     sentence = train_data[i].count("\n")
     average.append(len(token)/sentence)
     s = train_data[i].split("\n")
     m = len(nltk.word_tokenize(s[0]))
     for j in range(1, len(s)):
         if (len(nltk.word_tokenize(s[j])) > m):
             m = len(nltk.word_tokenize(s[j]))
     max.append(m)
average=np.array(average)
average.shape=(len(train_data),1)
max=np.array(max)
max.shape=(len(train_data),1)
tokens=np.c_[average,max]
X_train=sparse.hstack((X_train,tokens))

#feature4:average and maximum number of characters per word
characters=[]
max=[]
delset = string.punctuation
delset=delset+"\n"+"\r"
for i in range(0,len(train_data)):
    word=train_data[i].split(" ")
    c=len(word[0].translate(None, delset))
    m=c
    for j in range(1, len(word)):
         character=word[j].translate(None, delset)
         c=c+len(character)
         if (len(character) > m):
             m = len(character)
    characters.append(c/len(word))
    max.append(m)
characters=np.array(characters)
characters.shape=(len(train_data),1)
max=np.array(max)
max.shape=(len(train_data),1)
characters=np.c_[characters,max]
X_train=sparse.hstack((X_train,characters))

#feature5:average number of syllables per word
syllables=[]
for i in range(0,len(train_data)):
    word=train_data[i].split(" ")
    num = 0
    for j in range(0, len(word)):
        str = word[j].lower()
        num += str.count("a")
        num += str.count("i")
        num += str.count("u")
        num += str.count("o")
        num += str.count("e")
    syllables.append(num/len(word))
syllables=np.array(syllables)
syllables.shape=(len(train_data),1)
X_train=sparse.hstack((X_train,syllables))

#feature6: the Flesch-Kincaid score=0.39*AvgNumberWordsPerSentence+11.80*AvgNumberSyllablesPerWord-15.59
fk=[]
for i in range(0,len(train_data)):
    fk.append(0.39*average[i]+11.80*syllables[i])
fk=np.array(fk)
fk.shape=(len(train_data),1)
X_train=sparse.hstack((X_train,fk))


#feature7: SMOG 3+sqrt(number of polysyllable words in 30 sentences)
def ave_short(data):
    sen_length = 0
    sen_num = 0
    for i in range(0, len(data)):
        if (len(data[i].split("\n")) < 30):
            sen_num += 1
            sen_length += len(data[i].split("\n"))
    return sen_length / sen_num
def isPoly(word):
    num = 0
    str = word.lower()
    num += str.count("a")
    num += str.count("i")
    num += str.count("u")
    num += str.count("o")
    num += str.count("e")
    if (num >= 3):
        return 1
    else:
        return 0
def SMOG(text):
    conversion = [[6, 5], [12, 6], [20, 7], [30, 8], [42, 9], [56, 10], [72, 11], [90, 12], [110, 13], [132, 14],
                  [156, 15], [182, 16], [210, 17], [240, 18]]
    result = []
    for i in range(0, len(text)):
        sentence = text[i].split("\n")
        num = 0
        smog = 0
        if (len(sentence) >= 30):
            for i in range(0, 10):
                word = sentence[i].split(" ")
                for j in range(0, len(word)):
                    if (isPoly(word[j]) == 1):
                        num += 1
            for i in range(len(sentence) // 2 - 5, len(sentence) // 2 + 5):
                word = sentence[i].split(" ")
                for j in range(0, len(word)):
                    if (isPoly(word[j]) == 1):
                        num += 1

            for i in range(len(sentence) - 10, len(sentence)):
                word = sentence[i].split(" ")
                for j in range(0, len(word)):
                    if (isPoly(word[j]) == 1):
                        num += 1
            smog = 3 + math.sqrt(num)
        else:
            words = text[i].split(" ")
            for i in range(0, len(words)):
                if (isPoly(words[i]) == 1):
                    num += 1;
            sentence = len(sentence)
            average = num / sentence
            num += average * ave_short(text)
            j = 0
            while (j < len(conversion)):
                if (num <= conversion[j][0]):
                    smog = conversion[j][1]
                    break
                else:
                    j = j + 1
        result.append(smog)
    return result
smog=SMOG(train_data)
smog=np.array(smog)
smog.shape=(len(train_data),1)
X_train=sparse.hstack((X_train,smog))

#feature8: CTTR:divide the types by the square root of two times the tokens
def isExist(word, list):
    exist = 0
    for i in range(0, len(list)):
        if (list[i] == word):
            exist = 1
            break
    return exist
def CTTR(data):
    result = []
    delset = string.punctuation
    delset = delset + "\n" + "\r"
    for i in range(0, len(data)):
        text = data[i].translate(None, delset)
        type_list = []
        type = 0
        words = text.split(" ")
        for j in range(0, len(words)):
            words[j] = words[j].lower()
            if (isExist(words[j], type_list) == 0):
                type_list.append(words[j])
                type += 1
        cttr = type / math.sqrt(len(words))
        result.append(cttr)
    return result
cttr=CTTR(train_data)
cttr=np.array(cttr)
cttr.shape=(len(train_data),1)
X_train=sparse.hstack((X_train,cttr))


#feature9:POS based lexical variation and lexical density
def isTag(word,tag):
    if(tag=="noun"):
        if(word.find("NN")!=-1):
            return 1
    elif(tag=="verb"):
        if(word.find("VB")!=-1):
            return 1
    elif(tag=="adj"):
        if (word.find("JJ") != -1 or word.find("CD") != -1 or word.find("OD") != -1):
            return 1
    elif (tag == "adv"):
        if (word.find("RB") != -1):
            return 1
    elif (tag == "pre"):
        if (word.find("IN") != -1):
            return 1
    return 0
def POS(text):
    sentences=text.split("\n")
    tag_list=[]
    type=0
    num=0
    for i in range(0,len(sentences)):
        tokens = nltk.word_tokenize(sentences[i])
        tagged = nltk.pos_tag(tokens)
        for j in range(0, len(tagged)):
            word=tagged[j][1]
            if(isTag(word,"noun")==1 ):
                num+=1
                if(isExist("noun",tag_list)==0):
                    tag_list.append("noun")
                    type+=1
            elif(isTag(word,"verb")==1):
                num += 1
                if (isExist("verb", tag_list) == 0):
                    tag_list.append("verb")
                    type += 1
            elif (isTag(word, "adj") == 1):
                num += 1
                if (isExist("adj", tag_list) == 0):
                    tag_list.append("adj")
                    type += 1
            elif (isTag(word, "adv") == 1):
                num += 1
                if (isExist("adv", tag_list) == 0):
                    tag_list.append("adv")
                    type += 1
            elif (isTag(word, "pre") == 1):
                num += 1
                if (isExist("pre", tag_list) == 0):
                    tag_list.append("pre")
                    type += 1
    variation=type/num
    density=num/len(nltk.word_tokenize(text))
    return variation*density
pos=[]
for i in range(0,len(train_data)):
    pos.append(POS(train_data[i]))
pos=np.array(pos)
pos.shape=(len(pos),1)
X_train=sparse.hstack((X_train,pos))

#feature10:average number of noun,verb,adjective,adverb, prepositional phrases per sentence
def TAG(text):
    sentences=text.split("\n")
    noun = 0
    verb = 0
    adj = 0
    adv = 0
    pre = 0
    for i in range(0,len(sentences)):
        sentence=sentences[i]
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        for j in range(0,len(tagged)):
            word=tagged[j][1]
            if(isTag(word,"noun")==1):
                noun+=1
            elif(isTag(word,"verb")==1):
                verb+=1
            elif(isTag(word,"adj")==1):
                adj+=1
            elif(isTag(word,"adv")==1):
                adv+=1
            elif(isTag(word,"pre")==1):
                pre+=1
    result=[noun,verb,adj,adv,pre]
    return result
tag=[]
for i in range(0,len(train_data)):
    result=TAG(train_data[i])
    sentences=len(train_data[i].split("\n"))
    for j in range(0,len(result)):
        result[j]/=sentences
    tag.append(result)
tag=np.array(tag)
X_train=sparse.hstack((X_train,tag))

print X_train.shape
#a linear classifier
clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42).fit(X_train, y_train)
# clf=svm.SVC(C=1).fit(X_train, y_train)

#test
#feature1:tf-idf
X_new_counts = count_vect.transform(test_data)
X_test = tf_transformer.transform(X_new_counts)

#feature2:number of sentences per text
sentences=[]
for i in range(0,len(test_data)):
     sentence=test_data[i].count("\n")
     sentences.append(sentence)
sentences=np.array(sentences)
sentences.shape=(len(test_data),1)
X_test=sparse.hstack((X_test,sentences))

#feature3:average and maximum number of tokens per sentence
average=[]
max=[]
for i in range(0,len(test_data)):
     token=nltk.word_tokenize(test_data[i])
     sentence = test_data[i].count("\n")
     average.append(len(token)/sentence)
     s = test_data[i].split("\n")
     m = len(nltk.word_tokenize(s[0]))
     for j in range(1, len(s)):
         if (len(nltk.word_tokenize(s[j])) > m):
             m = len(nltk.word_tokenize(s[j]))
     max.append(m)
average=np.array(average)
average.shape=(len(test_data),1)
max=np.array(max)
max.shape=(len(test_data),1)
tokens=np.c_[average,max]
X_test=sparse.hstack((X_test,tokens))

#feature4:average and maximum number of characters per word
characters=[]
max=[]
delset = string.punctuation
delset=delset+"\n"+"\r"
for i in range(0,len(test_data)):
    word=test_data[i].split(" ")
    c=len(word[0].translate(None, delset))
    m=c
    for j in range(1, len(word)):
         character=word[j].translate(None, delset)
         c=c+len(character)
         if (len(character) > m):
             m = len(character)
    characters.append(c/len(word))
    max.append(m)
characters=np.array(characters)
characters.shape=(len(test_data),1)
max=np.array(max)
max.shape=(len(test_data),1)
characters=np.c_[characters,max]
X_test=sparse.hstack((X_test,characters))

#feature5:average number of syllables per word
syllables=[]
for i in range(0,len(test_data)):
    word=test_data[i].split(" ")
    num = 0
    for j in range(0, len(word)):
        str = word[j].lower()
        num += str.count("a")
        num += str.count("i")
        num += str.count("u")
        num += str.count("o")
        num += str.count("e")
    syllables.append(num/len(word))
syllables=np.array(syllables)
syllables.shape=(len(test_data),1)
X_test=sparse.hstack((X_test,syllables))

#feature6: the Flesch-Kincaid score=0.39*AvgNumberWordsPerSentence+11.80*AvgNumberSyllablesPerWord-15.59
fk=[]
for i in range(0,len(test_data)):
    fk.append(0.39*average[i]+11.80*syllables[i])
fk=np.array(fk)
fk.shape=(len(test_data),1)
X_test=sparse.hstack((X_test,fk))

#feature7: SMOG 3+sqrt(number of polysyllable words in 30 sentences)
smog=SMOG(test_data)
smog=np.array(smog)
smog.shape=(len(test_data),1)
X_test=sparse.hstack((X_test,smog))

#feature8: CTTR:divide the types by the square root of two times the tokens
cttr=CTTR(test_data)
cttr=np.array(cttr)
cttr.shape=(len(test_data),1)
X_test=sparse.hstack((X_test,cttr))

#feature9:POS based lexical variation and lexical density
pos=[]
for i in range(0,len(test_data)):
    pos.append(POS(test_data[i]))
pos=np.array(pos)
pos.shape=(len(pos),1)
X_test=sparse.hstack((X_test,pos))

#feature10:average number of noun,verb,adjective,adverb, prepositional phrases per sentence
tag=[]
for i in range(0,len(test_data)):
    result=TAG(test_data[i])
    sentences=len(test_data[i].split("\n"))
    for j in range(0,len(result)):
        result[j]/=sentences
    tag.append(result)
tag=np.array(tag)
X_test=sparse.hstack((X_test,tag))

print X_test.shape

predicted = clf.predict(X_test)

#evaluation
print np.mean(predicted == y_test)
names=["1","2","3","4"]
print(metrics.classification_report(y_test, predicted,target_names=names))

# #gridsearch
# text_clf = Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                                            alpha=1e-3, n_iter=5, random_state=42)),
# ])
#
#
# from sklearn.model_selection import GridSearchCV
# parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
#               'tfidf__use_idf': (True, False),
#               'clf__alpha': (1e-2, 1e-3),
# }
# gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
#
# gs_clf = gs_clf.fit(train_data, y_train)
# print gs_clf.best_score_
#
# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))