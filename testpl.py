from numpy import *
import math
import matplotlib.pyplot as plt



def read(fn):
    f = open(fn,'r')
    e = []
    dev = []
    test = []
    
    for line in f:
        if 'epoch: ' in line:
            lines = line.strip().split(' / ')
            e.append(int(lines[0].strip().replace('epoch: ','').strip()))
            dev.append(float(lines[1].strip().replace('dev_acc: ','').strip()))
            test.append(float(lines[2].strip().replace('test_acc: ','').strip()))
    
    f.close()
    return (e,test)
        

e, static = read('cnn-kim/result/static_eng')
plt.plot(e, static, 'b', label="Static") # plotting t, a separately 

e, nonstatic = read('cnn-kim/result/nonstatic_eng')
plt.plot(e, nonstatic, 'r', label="Non static") # plotting t, a separately 

e, rand = read('cnn-kim/result/rand_eng')
plt.plot(e, rand, 'y', label="Random") # plotting t, a separately 

e, multichannel = read('cnn-kim/result/multichannel_eng')
plt.plot(e, multichannel, 'g', label="Multichannel") # plotting t, a separately 

#plt.plot(e, dev, 'b', label="Development set") # plotting t, a separately 
#plt.plot(e, test, 'r', label="Test set") # plotting t, b separately 
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.5), fancybox=True, shadow=True, ncol=5)
plt.title('English static word embedding')
plt.show()
