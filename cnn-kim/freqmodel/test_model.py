import pdb
from os.path import join
from gensim.models.keyedvectors import KeyedVectors

datafolder = '../../data'
word_freq_file = join(datafolder, 'freqdict', 'english-word-byfreq.txt')
word2vec_file = join(datafolder, 'word2vec', 'eng.model.bin')

word_vectors = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)

sentence_list = [
    "He resembles his father in his way of talking.",
    "My father died of a subarachnoid hemorrhage when I was fourteen.",
    "I just found out that my dad is not my biological father.",
    "I'm sure your father is very proud of you.",
    "That is the girl whose father is a doctor.",
    "An uncle is a brother of your father or your mother.",
    "He's my uncle, because my father is his brother.",
    "My mother's father is my maternal grandfather.",
    "He named his son John after his own father.",
    "Both my father and my brother are fond of gambling.",
    "Dan was a good husband and a good father.",
    "The father asked for revenge against the man who deflowered his daughter.",
    "I don't know about João, but Maria lost her father when she was young.",
    "Father of the House of Commons",
    "father of Vietnamese poetry",
    "the Holy Father",
    "the wish is father to the thought"]

word = "father"

result = dict()
for i, sentence in enumerate(sentence_list):
    distance = word_vectors.wmdistance(word, sentence)
    result[i] = distance

import operator
sorted_result = sorted(result.items(), key=operator.itemgetter(1))

for t in sorted_result:
    print (sentence_list[t[0]], t[1])

