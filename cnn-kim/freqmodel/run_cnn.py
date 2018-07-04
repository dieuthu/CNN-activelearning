import model as rlc
from os.path import join
from preprocess import read_and_split_data


datafolder = '../../data/'
datalightfolder = '../../datalight/'
data = read_and_split_data(join(datalightfolder, 'txt', 'English_dataset.txt'))

word_freq_file = join(datafolder, 'freqdict', 'english-word-byfreq.txt')
word2vec_file = 'GoogleNews-vectors-negative300.bin'
#word2vec_file = join(datafolder, 'word2vec', 'eng.model.bin')

preprocessor = rlc.Preprocessor(word2vec_file, word_freq_file, topwords_as_vocab=False)
cls = rlc.ReadlevelClassifier(preprocessor, useGPU=False)
cls.cuda()


need_train = True
if need_train:
    cls.fit_sentavg(data['train_x'], data['train_y'],
            data['dev_x'], data['dev_y'],
            data['test_x'], data['test_y'])
    rlc.ReadlevelClassifier.save_model(cls, 'model.pkl')
else: cls = rlc.ReadlevelClassifier.load_model('model.pkl')

print ('model loaded/trained')
cls.cpu()
print (cls.classes)
print (cls.predict(["I am a student ! I like play games", ""]))
print (cls.predict(["This is a  an . !"]))
print (cls.predict(["This is a  kajfkdjakfdja  kdajkfjk very difficult hospital virus !"]))
print (cls.predict(["It is only in the study of man himself that the major social sciences have substituted the study of one local variation , that of Western civilization .  Anthropology was by definition impossible , as long as these distinctions between ourselves and the primitive , ourselves and the barbarian , ourselves and the pagan , held sway over people 's minds .  It was necessary first to arrive at that degree of sophistication where we no longer set our own belief against our neighbour 's superstition .  It was necessary to recognize that these institutions which are based on the same premises , let us say the supernatural , must be considered together , our own among the rest .  RUTH BENEDICT Patterns of Culture"]))
