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
import numpy as np
import torchvision.utils as vutils
from torchvision.utils import save_image
import random
import os
import shutil
import pdb


# In[2]:


# Initialization
num_channels = 1
num_classes = 10
latent_size = 100
labeled_rate = 0.1
num_epochs = 1000
image_size = 32
batch_size = 128
epsilon = 1e-8 # used to avoid NAN loss
generator_frequency = 5
discriminator_frequency = 1
dropout_rate = 0.25

# Initialize parameters
lr = 1e-5
b1 = 0.5 # adam: decay of first order momentum of gradient
b2 = 0.999 # adam: decay of first order momentum of gradient

log_path = './SSL_GAN_log.csv'
model_path ='./SSL_GAN_model.ckpt'

os.makedirs('images', exist_ok=True)


# In[3]:


DATA_FOLDER = './torch_data/MNIST'


# In[4]:


# Create Dataset
class MnistDataset(Dataset):
    def __init__(self, image_size, split):
        self.split = split
        self.mnist_dataset = self._create_dataset(image_size, split)
        self.label_mask = self._create_label_mask()
        
    def _create_dataset(self, image_size, split):
        compose = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        out_dir = '{}/dataset'.format(DATA_FOLDER)
        
        if self.split == 'train':
            return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
        else:
            return datasets.MNIST(root=out_dir, train=False, transform=compose, download=True)
        
    def _one_hot(self, y):
        label = y.item()
        label_onehot = np.zeros(num_classes + 1)
        label_onehot[label] = 1
        return label_onehot
    
    def _create_label_mask(self):
        if self.split == 'train':
            l = len(self.mnist_dataset)
            label_mask = np.zeros(l)
            masked_len = int(labeled_rate * l)
            label_mask[0:masked_len] = 1
            np.random.shuffle(label_mask)
            label_mask = torch.LongTensor(label_mask)
            return label_mask
        return None

    def __getitem__(self, idx):
        data, label = self.mnist_dataset.__getitem__(idx)
        label_onehot = self._one_hot(label)
        if self.split == 'train':
            return data, label, label_onehot, self.label_mask[idx]
        return data, label

    def __len__(self):
        return len(self.mnist_dataset)


# In[5]:


# Get dataloaders
def get_loader(image_size, batch_size):
    #num_workers = 2

    mnist_train = MnistDataset(image_size=image_size, split='train')
    mnist_test = MnistDataset(image_size=image_size, split='test')

    train_loader = DataLoader(
        dataset=mnist_train,
        batch_size=batch_size,
        shuffle=True
        #num_workers=num_workers
    )

    test_loader = DataLoader(
        dataset=mnist_test,
        batch_size=batch_size,
        shuffle=True
        #num_workers=num_workers
    )

    return train_loader, test_loader


# In[6]:


def initializer(m):
    # Run xavier on all weights and zero all biases
    if hasattr(m, 'weight'):
        if m.weight.ndimension() > 1:
            xavier_uniform_(m.weight.data)
    if hasattr(m, 'bias'):
        m.bias.data.zero_()


# In[7]:


class DiscriminatorNet(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
          
        d = 16
        
        # Conv operations
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=d*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=d*2, out_channels=d*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=d*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=d*4, out_channels=d*8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=d*8, out_channels=d*8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Linear 
        self.linear = nn.Linear(in_features=d*8, out_features=(num_classes + 1))
        self.softmax = nn.Softmax(dim=1)
        self.apply(initializer)
        
    def forward(self, x):
        # Convolutional Operations
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Linear
        flatten = x.view(x.size(0), -1)
        linear = self.linear(flatten)
        prob = self.softmax(linear)
        return flatten, linear, prob


# In[8]:


class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        
        d = 16
        
        # Conv operations
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=d*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d*8),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=d*8, out_channels=d*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=d*4),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=d*4, out_channels=d*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=d*2),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=d*2, out_channels=d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=d),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=d, out_channels=num_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.apply(initializer)
        
    def forward(self, x):
        
        x = x.view(-1, x.size(1), 1, 1)
        
        # Deconvolutional Operations
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        
        return x


# In[9]:


def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): 
        return n.cuda() 
    return n


# In[10]:


# Models
discriminator = DiscriminatorNet()
generator = GeneratorNet()

# Data Loader
train_loader, test_loader = get_loader(image_size, batch_size)

# Optimizers
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))

# Loss
bce_loss = nn.BCEWithLogitsLoss()
cross_loss = nn.CrossEntropyLoss(reduction='none')

if torch.cuda.is_available():
    discriminator = discriminator.cuda()
    generator = generator.cuda()
    bce_loss = bce_loss.cuda()
    cross_loss = cross_loss.cuda()


# In[11]:


# Visualize Data
def plot_fake_data(data, grid_size = [5, 5]):
    _, axes = plt.subplots(figsize = grid_size, nrows = grid_size[0], ncols = grid_size[1],
                           sharey = True, sharex = True)

    size = grid_size[0] * grid_size[1]
    index = np.int_(np.random.uniform(0, data.shape[0], size = (size)))

    figs = data[index].reshape(-1, image_size, image_size)

    for idx, ax in enumerate(axes.flatten()):
        ax.axis('off')
        ax.imshow(figs[idx], cmap = 'gray')
    plt.tight_layout()
    plt.show()


# In[12]:


def train_discriminator(optimizer_D, b_size, img, label, label_mask, epsilon):
    
    # Generate Fake Image
    z = noise(b_size)
    fake_img = generator(z)

    # Discriminator outputs for real and fake
    d_real_flatten, d_real_linear, d_real_prob = discriminator(img.detach())
    d_fake_flatten, d_fake_linear, d_fake_prob = discriminator(fake_img)
    
    optimizer_D.zero_grad()
        
    # Supervised Loss
    supervised_loss = cross_loss(d_real_linear, label)
#   d_class_loss_entropy = - torch.sum(label_onehot.float() * torch.log(d_real_prob), dim=1)

    masked_supervised_loss = torch.mul(label_mask, supervised_loss)
    delim = torch.Tensor([1.0])
    if torch.cuda.is_available():
        delim = delim.cuda()
    mask_sum = torch.max(delim, torch.sum(label_mask))
    d_class_loss = torch.sum(label_mask * masked_supervised_loss) / mask_sum

    # Unsupervised (GAN) Loss
    # data is real
    prob_real_is_real = 1.0 - d_real_prob[:, -1] + epsilon
    tmp_log = torch.log(prob_real_is_real)
    d_real_loss = -1.0 * torch.mean(tmp_log)

    # data is fake
    prob_fake_is_fake = d_fake_prob[:, -1] + epsilon
    tmp_log = torch.log(prob_fake_is_fake)
    d_fake_loss = -1.0 * torch.mean(tmp_log)

    # loss and weight update
    d_loss = d_class_loss + d_real_loss + d_fake_loss
    d_loss.backward(retain_graph=True)
    optimizer_D.step()
    
    # Accuracy
    _, predicted = torch.max(d_real_prob[:, :-1], dim=1)
    correct_batch = torch.sum(torch.eq(predicted, label))
    batch_accuracy = correct_batch.item()/float(b_size)
    
    return d_loss, batch_accuracy


# In[13]:


def train_generator(optimizer_G, b_size, epsilon):
    
    # Generate Fake Image
    z = noise(b_size)
    fake_img = generator(z)

    # Discriminator outputs for real and fake
    d_real_flatten, d_real_linear, d_real_prob = discriminator(img.detach())
    d_fake_flatten, d_fake_linear, d_fake_prob = discriminator(fake_img)
    
    optimizer_G.zero_grad()
        
    # fake data is mistaken to be real
    prob_fake_is_real = 1.0 - d_fake_prob[:, -1] + epsilon
    tmp_log =  torch.log(prob_fake_is_real)
    g_fake_loss = -1.0 * torch.mean(tmp_log)

    # Feature Maching
    tmp1 = torch.mean(d_real_flatten, dim = 0)
    tmp2 = torch.mean(d_fake_flatten, dim = 0)
    diff = tmp1 - tmp2
    g_feature_loss = torch.mean(torch.mul(diff, diff))

    # Loss and weight update
    g_loss = g_fake_loss + g_feature_loss
    g_loss.backward()
    optimizer_G.step()

    return g_loss, fake_img


# In[14]:


'''
Start Training
'''
generator.train()
discriminator.train()

for epoch in range(num_epochs):
    total_accuracy = 0
    G_loss = 0
    D_loss = 0
    
    for i, data in enumerate(train_loader):
        
        img, label, label_onehot, label_mask = data
        label_mask = label_mask.float()
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
            label_onehot = label_onehot.cuda()
            label_mask = label_mask.cuda()
            
        b_size = img.size(0)
        
        ################### Discriminator ####################
        batch_d_loss = 0
        batch_accuracy = 0
        
        for d_i in range(discriminator_frequency):
            d_loss, d_accuracy = train_discriminator(optimizer_D, b_size, img, label, label_mask, epsilon)
            batch_d_loss += d_loss.item()  
            batch_accuracy += d_accuracy
            
        batch_d_loss = batch_d_loss/float(discriminator_frequency)
        batch_accuracy = batch_accuracy/float(discriminator_frequency)
        
        ################### Generator ####################
        batch_g_loss = 0
        for g_i in range(generator_frequency):
            g_loss, fake_img = train_generator(optimizer_G, b_size, epsilon)
            batch_g_loss += g_loss.item()
        batch_g_loss = batch_g_loss/float(generator_frequency)
       
        total_accuracy += batch_accuracy
        D_loss += batch_d_loss
        G_loss += batch_g_loss
        
        if i%b_size == b_size-1:
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]" % (epoch, num_epochs, i, 
                                       len(train_loader), d_loss.item(), 100 * batch_accuracy, g_loss.item()))

        
    # Print Epoch results
    total_accuracy = total_accuracy/float(i+1)
    avg_D_loss = D_loss/float(i+1)
    avg_G_loss = G_loss/float(i+1)
    
    print('--------------------------------------------------------------------')
    print("===> [Epoch %d/%d] [Avg D loss: %f, avg acc: %d%%] [Avg G loss: %f]" % (epoch, num_epochs, 
                                                        avg_D_loss, 100 * total_accuracy, avg_G_loss))
    print('--------------------------------------------------------------------')
    
    # Save Images
    save_image(fake_img[:25], 'images/epoch_%d_batch_%d.png' % (epoch, i), nrow=5, normalize=True)


# In[ ]:





# In[ ]:




