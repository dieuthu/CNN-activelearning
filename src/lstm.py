import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable

# (1, 0) => target labels 0+2
# (0, 1) => target labels 1
# (1, 1) => target labels 3
train = [(1,0), (0,1), (1,1)]
labels = [0, 1, 3]


time_steps = 10
batch_size = 3
in_size = 5
classes_no = 7



for i in range(10000):
    category = (np.random.choice([0, 1]), np.random.choice([0, 1]))
    if category == (1, 0):
        train.append([np.random.uniform(0.1, 1), 0])
        labels.append([1, 0, 1])
    if category == (0, 1):
        train.append([0, np.random.uniform(0.1, 1)])
        labels.append([0, 1, 0])
    if category == (0, 0):
        train.append([np.random.uniform(0.1, 1), np.random.uniform(0.1, 1)])
        labels.append([0, 0, 1])

class _classifier(nn.Module):
    def __init__(self, nlabel):
        super(_classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, nlabel),
        )

    def forward(self, input):
        return self.main(input)

nlabel = len(labels[0]) # => 3
classifier = _classifier(nlabel)

optimizer = optim.Adam(classifier.parameters())
criterion = nn.MultiLabelSoftMarginLoss()

epochs = 5
for epoch in range(epochs):
    losses = []
    for i, sample in enumerate(train):
        inputv = Variable(torch.FloatTensor(sample)).view(1, -1)
        labelsv = Variable(torch.FloatTensor(labels[i])).view(1, -1)
        
        output = classifier(inputv)
        loss = criterion(output, labelsv)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.mean())
    print('[%d/%d] Loss: %.3f' % (epoch+1, epochs, np.mean(losses)))
	
######	



model = nn.LSTM(in_size, classes_no, 2)
input_seq = Variable(torch.randn(time_steps, batch_size, in_size))
output_seq, _ = model(input_seq)
last_output = output_seq[-1]

loss = nn.CrossEntropyLoss()
target = Variable(torch.LongTensor(batch_size).random_(0, classes_no-1))
err = loss(last_output, target)
err.backward()



