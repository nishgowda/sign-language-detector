from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

import sys
from model import CNN

signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', 
         '6': 'G', '7': 'H', '8': 'I', '10': 'K', '11': 'L', '12': 'M', 
         '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R', '18': 'S', 
         '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y' }

class SignsLanguageDataset(Dataset):
    def __init__(self, train = True, transform = None):
        self.train = train
        self.transform = transform
        
        # uses standard mnist sign language translation testing and training data
        if self.train == True:
            self.signs_lang_dataset = pd.read_csv('data/input/sign_mnist_train.csv')
        else:
            self.signs_lang_dataset = pd.read_csv('data/input/sign_mnist_test.csv')
            
        self.X_set = self.signs_lang_dataset.iloc[:, 1:].values
        self.y_set = self.signs_lang_dataset.iloc[:, 0].values
        
        self.X_set = np.reshape(self.X_set, (self.X_set.shape[0], 1, 28, 28)) / 255
        self.y_set = np.array(self.y_set)
            
    def __getitem__(self, index):
        image = self.X_set[index, :, :]
        label = self.y_set[index]
        sample = {'image_sign': image, 'label': label}
        return sample
        
    def __len__(self):
        return self.X_set.__len__()

def train(model, optimizer, epoch, device, train_loader, log_interval):
    model.train()

    for batch_idx, data in enumerate(train_loader):
        img = data['image_sign']
        img = img.type(torch.FloatTensor).to(device)
        
        target = data['label']
        target = target.type(torch.LongTensor).to(device)
        
        optimizer.zero_grad()
        
        output = model(img)
        loss = F.nll_loss(output, target)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            img = data['image_sign']
            img = img.type(torch.FloatTensor).to(device)
            target = data['label']
            target = target.type(torch.LongTensor).to(device)
            
            output = model(img)
            test_loss += F.nll_loss(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

batch_size_train = 5
batch_size_test = 4

dataset_train = SignsLanguageDataset(train=True)
dataset_test = SignsLanguageDataset(train=False)
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size_train)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size_test)

torch.manual_seed(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
num_epochs = 7

model = CNN()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.2, weight_decay=0.002)

log_interval = 27455

for epoch in range(1, num_epochs + 1):
    train(model, optimizer, epoch, device, train_loader, log_interval)
    test(model, device, test_loader)

model_name = sys.argv[1]

model_path = f"models/{model_name}"
torch.save(model.state_dict(), model_path)
print(f"Saved model to {model_path}")

