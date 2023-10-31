###################################### Q 4.2 ##############################################

import numpy as np

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import torch

import time



transform = transforms.Compose([transforms.ToTensor(),

                                transforms.Normalize((0.5,), (0.5,)),

                                ])



input_size = 784

hidden_layer_size = 300

output_size = 10

losses = []

accuracies = []


class CrossNeuralModel():

    def __init__(self, sizes, epochs=20, alpha=0.01):

        self.sizes = sizes

        self.epochs = epochs

        self.alpha = alpha

        self.init_params()

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

    def softmax(self, x):

        exes = np.exp(x)

        deno = np.sum(exes, axis=1)

        deno.resize(exes.shape[0], 1)

        return exes / deno

    def init_params(self):

        input_layer = int(self.sizes[0])

        hidden_1 = int(self.sizes[1])

        output_layer = int(self.sizes[2])

        # Random initialization of weights between -1 and 1

        self.w1 = np.random.uniform(low=-1, high=1, size=(input_layer, hidden_1))

        self.w2 = np.random.uniform(low=-1, high=1, size=(hidden_1, output_layer))

        # Zero initialization of weights

        # self.w1 = np.zeros((input_layer, hidden_1))

        # self.w2 = np.zeros((hidden_1, output_layer))

    def forward(self, inputs):

        inputs = inputs.numpy()

        self.linear_1 = inputs.dot(self.w1)

        self.out1 = self.sigmoid(self.linear_1)

        self.linear2 = self.out1.dot(self.w2)

        self.out2 = self.softmax(self.linear2)

        return self.out2

    def backward(self, x_train, y_train, output):

        x_train = x_train.numpy()

        y_train = y_train.numpy()

        batch_size = y_train.shape[0]

        d_loss = output - y_train

        delta_w2 = (1. / batch_size) * np.matmul(self.out1.T, d_loss)

        d_out_1 = np.matmul(d_loss, self.w2.T)

        d_linear_1 = d_out_1 * self.sigmoid(self.linear_1) * (1 - self.sigmoid(self.linear_1))


        delta_w1 = (1. / batch_size) * np.matmul(x_train.T, d_linear_1)

        return delta_w1, delta_w2

    def update_weights(self, w1_update, w2_update):

        self.w1 -= self.alpha * w1_update

        self.w2 -= self.alpha * w2_update

    def calculate_loss(self, y, y_hat):

        batch_size = y.shape[0]

        y = y.numpy()

        loss = np.sum(np.multiply(y, np.log(y_hat)))

        loss = -(1. / batch_size) * loss

        return loss

    def calculate_metrics(self, data_loader):

        losses = []

        correct = 0

        total = 0

        for i, data in enumerate(data_loader):
            x, y = data

            y_onehot = torch.zeros(y.shape[0], 10)

            y_onehot[range(y_onehot.shape[0]), y] = 1

            flattened_input = x.view(-1, 28 * 28)

            output = self.forward(flattened_input)

            predicted = np.argmax(output, axis=1)

            correct += np.sum((predicted == y.numpy()))

            total += y.shape[0]


            loss = self.calculate_loss(y_onehot, output)

            losses.append(loss)


        return (correct / total), np.mean(np.array(losses))

    def train(self, train_loader, data_loader):

        start_time = time.time()

        global losses, accuracies

        for iteration in range(self.epochs):

            for i, data in enumerate(train_loader):
                x, y = data

                y_onehot = torch.zeros(y.shape[0], 10)

                y_onehot[range(y_onehot.shape[0]), y] = 1

                flat_input = x.view(-1, 28 * 28)

                output = self.forward(flat_input)

                w1_update, w2_update = self.backward(flat_input, y_onehot, output)

                self.update_weights(w1_update, w2_update)

            accuracy, loss = self.calculate_metrics(data_loader)

            losses.append(loss)

            accuracies.append(accuracy)

            print('Epoch: {0}, Test Error Percent: {1:.2f}, Loss: {2:.2f}'.format(

                iteration + 1, 100 - accuracy * 100, loss

            ))


if __name__ == '__main__':
    model = CrossNeuralModel(sizes=[784, 300, 10], epochs=50, alpha=0.001)
    bsize=32
    trainset = datasets.MNIST('./dataset/MNIST/', download=True, train=True, transform=transform)
    testset = datasets.MNIST('./dataset/MNIST/', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=bsize, shuffle=True)
    model.train(train_loader=trainloader, data_loader=dataloader)
    plt.xlabel('Epochs')
    plt.ylabel('Test Loss')
    plt.plot(losses)
    plt.show()




########################################### Q 4.3-4.4 ##############################################

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import optim
import numpy as np
# %matplotlib inline

bsize=32
rate=0.01
transform = transforms.Compose([transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.MNIST('data', train=True, transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bsize, shuffle=True)

testset = torchvision.datasets.MNIST('data', train=True, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=bsize, shuffle=True)

input_size = trainloader.dataset.train_data.shape[1] * trainloader.dataset.train_data.shape[2]
hidden_layers = [300]
output_size = 10

def init_weights(m):
  if type(m) == nn.Linear:
    torch.nn.init.uniform_(m.weight,-1.0,1.0)
    #torch.nn.init.zeros_(m.weight)


model = nn.Sequential(
    nn.Linear(input_size, hidden_layers[0]),
    nn.Sigmoid(),
    nn.Linear(hidden_layers[0], output_size),
    nn.LogSoftmax(dim=1)
)
model.apply(init_weights)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=rate)

epochs = 50
losses = []
for e in range(epochs):
    running_loss = 0
    for x, y in trainloader:
        
        x = x.view(x.shape[0], -1)
        
        # setting gradient to zeros
        optimizer.zero_grad()        
        output = model(x)
        loss = criterion(output, y)
        
        # backward propagation
        loss.backward()
        
        # update the gradient to new gradients
        optimizer.step()
        running_loss += loss.item()
    else:
        print("Epoch: ",e+1)
        print("Running loss: ",(running_loss/len(trainloader)))
        losses.append(running_loss/len(trainloader))


correct=0
with torch.no_grad():
  for images,labels in testloader:
    logps = model(images.view(images.shape[0], -1))
    output = torch.squeeze(logps)
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(labels.data.view_as(pred)).sum()
  print('\nAccuracy Percent: {}/{} ({:.0f})\n'.format(correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))
  print('\nTest Error Percent: ({:.0f})\n'.format(100 - 100. * correct / len(testloader.dataset)))  

plt.xlabel('Epochs')
plt.ylabel('Test Loss')
plt.plot(losses)
plt.show()
