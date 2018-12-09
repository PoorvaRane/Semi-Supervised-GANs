#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.utils import weight_norm
import numpy as np
import torchvision.utils as vutils
from torchvision.utils import save_image
import random
import os
import shutil
import pdb
from logger import Logger
from PIL import Image


# In[2]:


# Initialization
num_channels = 3
num_classes = 1
num_epochs = 300
image_size = 64
batch_size = 64
epsilon = 1e-8 # used to avoid NAN loss
logger = Logger('./logs')

# Initialize parameters
lr = 1e-5
b1 = 0.5 # adam: decay of first order momentum of gradient
b2 = 0.999 # adam: decay of first order momentum of gradient

model_path ='./baseline_64.tar'
graph_dir ='./result_graphs/'

# In[3]:


# Create Dataset
class TCGADataset(Dataset):
    def __init__(self, image_size, split):
        self.split = split
        self.tcga_dataset = self._create_dataset(image_size, split)
        self.patches, self.labels = self.tcga_dataset
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        
    def _create_dataset(self, image_size, split):
        data_dir = '/mys3bucket/patch_data'
        if self.split == 'train':
            data_dir = os.path.join(data_dir, 'train')
        else:
            data_dir = os.path.join(data_dir, 'dev')
        
        all_files = os.listdir(data_dir)
        images = []
        labels = []
        
        # Iterate over all files
        for file in all_files:
            if '.npz' not in file:
                continue
            file_path = os.path.join(data_dir, file)
            data = np.load(file_path)
            X = data['arr_0']
            y = data['arr_1']
            images.append(X)
            labels.append(y)                
            
        images = np.concatenate(images)
        labels = np.concatenate(labels) 
        #balance data
        cancer = np.count_nonzero(labels)
        noncancer = (labels.shape[0]-cancer)
        minimum = min(cancer,noncancer)
        sample_idxs_cancer = random.sample(list(np.where(labels == 1)[0]), minimum)
        sample_idxs_nocancer = random.sample(list(np.where(labels == 0)[0]), minimum)
        new_idxs = []
        new_idxs.extend(sample_idxs_cancer)
        new_idxs.extend(sample_idxs_nocancer)
        random.shuffle(new_idxs)
        images = images[new_idxs]
        labels = labels[new_idxs]
        # Print data statistics
        print("Total number of patches : ",labels.shape[0])
        c = 0
        nc = 0
        for l in labels:
            if l:
                c+=1
            else:
                nc+=1

        print("Cancerous patches : ",c )
        print("Non cancerous patches : ",nc )
                    
        return images, labels
    

    def __getitem__(self, idx):
        data, label = self.patches[idx], self.labels[idx]
        return self.transform(Image.fromarray(data)), label

    def __len__(self):
        return len(self.labels)


# In[4]:


# Get dataloaders
def get_loader(image_size, batch_size):
    num_workers = 2
    tcga_train = TCGADataset(image_size=image_size, split='train')
    tcga_dev = TCGADataset(image_size=image_size, split='dev')

    train_loader = DataLoader(
        dataset=tcga_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    dev_loader = DataLoader(
        dataset=tcga_dev,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_loader, dev_loader


# In[5]:


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.zero_()
        
        
def initializer(m):
    # Run xavier on all weights and zero all biases
    if hasattr(m, 'weight'):
        if m.weight.ndimension() > 1:
            xavier_uniform_(m.weight.data)

    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.zero_() 


# In[6]:


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
          
        dropout_rate = 0.5
        filter1 = 96
        filter2 = 192
        
        # Conv operations
        # CNNBlock 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=filter1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter1),
            nn.LeakyReLU(0.2),
#             nn.Conv2d(in_channels=filter1, out_channels=filter1, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(filter1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(filter2),
#             nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate)
        )
        
        # CNNBlock 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter2),
            nn.LeakyReLU(0.2),
#             nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(filter2),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(filter2),
#             nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate)
        )

        # CNNBlock 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filter2),
            nn.LeakyReLU(0.2),
#             nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(filter2),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(filter2),
#             nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate)
        )
        
        # CNNBlock 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(filter2),            
            nn.LeakyReLU(0.2),
#             nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(filter2),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(filter2),
#             nn.LeakyReLU(0.2)
        )
                
        # Linear 
        self.linear = nn.Sequential(
            nn.Linear(in_features=filter2, out_features=(num_classes))
        )
        self.apply(initializer)
        
    def forward(self, x):
        # Convolutional Operations
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Linear
        x = x.mean(dim=3).mean(dim=2)
        x = self.linear(x)
        x = F.sigmoid(x)
        
        return x


# In[7]:


# Initialize loss and model
criterion = nn.BCELoss()
model = Model()

# Data Loader
train_loader, dev_loader = get_loader(image_size, batch_size)

# Initialize weights
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2))

if torch.cuda.is_available():
    criterion.cuda()
    model = nn.DataParallel(model)
    model.cuda()


# In[8]:


# Training Function
def train(epoch, num_epochs, optimizer, criterion, dataloader, model):
    model.train()
    
    total_loss = 0
    total_acc = 0
    loader_len = len(dataloader)

    for i, data in enumerate(dataloader):
        
        optimizer.zero_grad()
        img, label = data
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        b_size = img.size(0)
    
        # Loss computation
        probs = model(img)
        probs = probs.squeeze()
        loss = criterion(probs, label.float())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        # Train Accuracy Computation
        compare = torch.FloatTensor([0.5])
        if torch.cuda.is_available:
            compare = compare.cuda()
            
        predicted = torch.ge(probs, compare)
        correct = torch.sum(torch.eq(predicted.long(), label))
        batch_accuracy = correct.item()/float(b_size)
        total_acc += batch_accuracy
        
        # Print stats
        if i%b_size == b_size-1:
            print("Train [Epoch %d/%d] [Batch %d/%d] [loss: %f, acc: %d%%]" % (epoch, num_epochs, i, 
                                       loader_len, loss.item(), 100 * batch_accuracy))
            
    total_loss = total_loss/float(i+1)
    total_acc = total_acc/float(i+1)
    return total_loss, total_acc


# In[9]:


# Testing Function
def test(epoch, num_epochs, criterion, dataloader, model, mode):
    model.eval()
    
    total_loss = 0
    total_acc = 0
    loader_len = len(dataloader)

    for i, data in enumerate(dataloader):
        
        img, label = data
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        b_size = img.size(0)
    
        # Loss computation
        probs = model(img)
        probs = probs.squeeze()
        loss = criterion(probs, label.float())
        total_loss += loss.item()
        
        # Train Accuracy Computation
        compare = torch.FloatTensor([0.5])
        if torch.cuda.is_available:
            compare = compare.cuda()
            
        predicted = torch.ge(probs, compare)
        correct = torch.sum(torch.eq(predicted.long(), label))
        batch_accuracy = correct.item()/float(b_size)
        total_acc += batch_accuracy
        
        # Print stats
        if i%b_size == b_size-1:
            print("%s [Epoch %d/%d] [Batch %d/%d] [loss: %f, acc: %d%%]" % (mode, epoch, num_epochs, i, 
                                       loader_len, loss.item(), 100 * batch_accuracy))
            
    total_loss = total_loss/float(i+1)
    total_acc = total_acc/float(i+1)
    return total_loss, total_acc


# In[10]:


def save_checkpoint(state, is_best):
    if is_best:
        torch.save(state, 'baseline.tar')


# In[11]:

def plot_graph(epoch, train, dev, mode):
    #pdb.set_trace()
    epoch_list = np.arange(epoch + 1)
    plt.plot(epoch_list, train)
    plt.plot(epoch_list, dev)
    
    if mode.lower() == 'accuracy':
        location = 'lower right'
    else:
        location = 'upper right'

    plt.legend(['Train ' +  mode, 'Dev ' + mode], loc=location)
    plt.xlabel('Epochs')
    plt_image_path = os.path.join(graph_dir, 'Baseline_64_' + mode.lower()[:4] + '_epoch_' + str(epoch))
    plt.savefig(plt_image_path)


'''
Call Train and Test and save best model
'''
best_valid_acc = 0.0
best_valid_loss = 99999.0
train_losses = []
val_losses = []
train_accuracy = []
val_accuracy = []

for epoch in range(num_epochs):
    train_loss, train_acc = train(epoch, num_epochs, optimizer, criterion, train_loader, model)
    print('------')
    valid_loss, valid_acc = test(epoch, num_epochs, criterion, dev_loader, model, 'Valid')
    
    print('--------------------------------------------------------------------')
    print("Train ===> [Epoch %d/%d] [Train loss: %f, train acc: %f%%]" % (epoch, num_epochs, 
                                                                          train_loss, 100 * train_acc))
    print("Valid ===> [Epoch %d/%d] [Valid loss: %f, valid acc: %f%%]" % (epoch, num_epochs, 
                                                                          valid_loss, 100 * valid_acc))
    print('--------------------------------------------------------------------')
      
    # Save best model
    is_best = valid_acc >= best_valid_acc
    save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    'best_acc' : valid_acc
    }, is_best)
    
    # Plot train/val graphs
    train_accuracy.append(train_acc)
    val_accuracy.append(valid_acc)
    plot_graph(epoch, train_accuracy, val_accuracy, 'Accuracy')
    train_losses.append(train_loss)
    val_losses.append(valid_loss)
    plot_graph(epoch, train_losses, val_losses, 'Loss')


# In[ ]:





# In[ ]:




