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
import argparse
import models
import utils
# from logger import Logger
from PIL import Image


# In[3]:


parser = argparse.ArgumentParser()
parser.add_argument('--num_channels', type=int, default=3, help='number of channels')
parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
parser.add_argument('--latent_size', type=int, default=100, help='latent size for noise vector')
parser.add_argument('--labeled_rate', type=float, default=0.1, help='ratio of labeled to unlabled samples')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon')
parser.add_argument('--generator_frequency', type=int, default=1, help='generator frequency')
parser.add_argument('--discriminator_frequency', type=int, default=1, help='discriminator frequency')
parser.add_argument('--lrD', type=float, default=1e-5, help='discriminator learning rate')
parser.add_argument('--lrG', type=float, default=1e-5, help='generator learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--b2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--param', type=str, default='32', help='parameter setting')
parser.add_argument('--mode', type=str, default='train', help='train or test the model')
args = parser.parse_args()

# Print args
print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')


# In[4]:


# logger = Logger('./logs')
image_dir = 'images_' + args.param
graph_dir = 'result_graphs'

os.makedirs(image_dir, exist_ok=True)
os.makedirs(image_dir + '_fixed', exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)


# In[10]:


# Models
if args.image_size == 32:
    discriminator = models.DiscriminatorNet(args)
    generator = models.GeneratorNet(args)
else:
    discriminator = models.DiscriminatorNet_64(args)
    generator = models.GeneratorNet_64(args)

# Data Loader
train_loader, dev_loader = utils.get_loader(args)

# Optimizers
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lrD, betas=(args.b1, args.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lrG, betas=(args.b1, args.b2))

# Loss
mse_loss = nn.MSELoss()
cross_loss = nn.CrossEntropyLoss(reduction='none')

if torch.cuda.is_available():
    discriminator = discriminator.cuda()
    generator = generator.cuda()
    discriminator = nn.DataParallel(discriminator)
    generator = nn.DataParallel(generator)
    mse_loss = mse_loss.cuda()
    cross_loss = cross_loss.cuda()


# In[12]:


def train_discriminator(optimizer_D, b_size, img, label, label_mask, epsilon):
    
    # Generate Fake Image
    z = utils.noise(b_size)
    fake_img = generator(z)

    # Discriminator outputs for real and fake
    d_real_flatten, d_real_linear, d_real_prob = discriminator(img)
    d_fake_flatten, d_fake_linear, d_fake_prob = discriminator(fake_img.detach())
    
    optimizer_D.zero_grad()
        
    # Supervised Loss
    supervised_loss = cross_loss(d_real_linear, label)

    masked_supervised_loss = torch.mul(label_mask, supervised_loss)
    delim = torch.Tensor([1.0])
    if torch.cuda.is_available():
        delim = delim.cuda()
    mask_sum = torch.max(delim, torch.sum(label_mask))
    d_class_loss = torch.sum(label_mask * masked_supervised_loss) / mask_sum

    # Unsupervised (GAN) Loss - Least Squares GAN Loss
    d_loss = 0.5 * (torch.mean((d_real_linear - 1)**2) +torch.mean(d_fake_linear**2))
    
    # loss and weight update
    d_loss = d_class_loss + d_loss
    d_loss.backward(retain_graph=True)
    optimizer_D.step()
    
    # Accuracy
    _, predicted = torch.max(d_real_prob[:, :-1], dim=1)
    correct_batch = torch.sum(torch.eq(predicted, label))
    batch_accuracy = correct_batch.item()/float(b_size)
    
    return d_loss, batch_accuracy


# In[ ]:


def test_discriminator(b_size, img, label):
    
    # Generate Fake Image
    z = utils.noise(b_size)
    fake_img = generator(z)

    # Discriminator outputs for real and fake
    d_real_flatten, d_real_linear, d_real_prob = discriminator(img)
    d_fake_flatten, d_fake_linear, d_fake_prob = discriminator(fake_img.detach())
    
    # Accuracy
    _, predicted = torch.max(d_real_prob[:, :-1], dim=1)
    correct_batch = torch.sum(torch.eq(predicted, label))
    batch_accuracy = correct_batch.item()/float(b_size)
    
    return batch_accuracy


# In[13]:


def train_generator(img, optimizer_G, b_size, epsilon):
    
    # Generate Fake Image
    z = utils.oise(b_size)
    fake_img = generator(z)

    # Discriminator outputs for real and fake
    d_real_flatten, d_real_linear, d_real_prob = discriminator(img)
    d_fake_flatten, d_fake_linear, d_fake_prob = discriminator(fake_img)
    
    optimizer_G.zero_grad()
    
    #Least squares loss for generator
    g_fake_loss = 0.5 * torch.mean((d_fake_linear - 1)**2)

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


# In[15]:

def training_module(epoch, train_loader):
    generator.train()
    discriminator.train()
    total_train_accuracy = 0
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
        
        for d_i in range(args.discriminator_frequency):
            d_loss, d_accuracy = train_discriminator(optimizer_D, b_size, img, label, label_mask, args.epsilon)
            batch_d_loss += d_loss.item()  
            batch_accuracy += d_accuracy
            
        batch_d_loss = batch_d_loss/float(args.discriminator_frequency)
        train_batch_accuracy = batch_accuracy/float(args.discriminator_frequency)
        
        ################### Generator ####################
        batch_g_loss = 0
        for g_i in range(args.generator_frequency):
            g_loss, fake_img = train_generator(img, optimizer_G, b_size, args.epsilon)
            batch_g_loss += g_loss.item()
        batch_g_loss = batch_g_loss/float(args.generator_frequency)
       
        total_train_accuracy += train_batch_accuracy
        D_loss += batch_d_loss
        G_loss += batch_g_loss    
        
        if i%b_size == b_size-1:
            print("Train [Epoch %d/%d] [Batch %d/%d] [D loss: %f, train acc: %.3f%%] [G loss: %f]" % (epoch, args.num_epochs,
                          i, len(train_loader), batch_d_loss, 100 * train_batch_accuracy, batch_g_loss))

    # Epoch Stats
    total_train_accuracy = total_train_accuracy/float(i+1)
    D_loss = D_loss/float(i+1)
    G_loss = G_loss/float(i+1)

    return total_train_accuracy, D_loss, G_loss, fake_img


def eval_module(dev_loader):
    generator.eval()
    discriminator.eval()
    total_dev_accuracy = 0

    for i, data in enumerate(dev_loader):
        
        img, label = data
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        
        b_size = img.size(0)
        dev_accuracy = test_discriminator(b_size, img, label)
        total_dev_accuracy += dev_accuracy
        
    # Epoch Stats
    total_dev_accuracy = total_dev_accuracy/float(i+1)
    return total_dev_accuracy


        
def main_module():
    # Fixed noise vector
    fixed_z = noise(64)
    train_acc_list = []
    dev_acc_list = []

    for epoch in range(args.num_epochs):

        # Training
        total_train_accuracy, D_loss, G_loss, fake_img = training_module(epoch, train_loader)
        # Evaluation 
        total_dev_accuracy = eval_module(dev_loader)

        # Save best model
        is_best = total_dev_accuracy >= total_train_accuracy

        utils.save_checkpoint({
        'epoch': epoch + 1,
        'dis_state_dict': discriminator.state_dict(),
        'optimizer_D' : optimizer_D.state_dict(),
        'gen_state_dict': generator.state_dict(),
        'optimizer_G' : optimizer_G.state_dict(),
        'dev_accuracy' : total_dev_accuracy,
        'train_accuracy' : total_train_accuracy,
        }, args, is_best)
        
        print('--------------------------------------------------------------------')
        print("===> [Epoch %d/%d] [Avg D loss: %f, avg train acc: %.3f%%, avg dev acc: %.3f%%] [Avg G loss: %f]" % (epoch, args.num_epochs, 
                                                      D_loss, 100 * total_train_accuracy, 100* total_dev_accuracy, G_loss))
        print('--------------------------------------------------------------------')
        
        # Save Images
        save_image(fake_img, image_dir + '/epoch_%d.png' % (epoch), nrow=8, normalize=True)
        # Save Fixed Images
        fixed_fake_img = generator(fixed_z)
        save_image(fixed_fake_img, image_dir + '_fixed' + '/epoch_%d.png' % (epoch), nrow=8, normalize=True)
        
        # Tensorboard logging 
#         utils.tensorboard_logging(epoch, G_loss, D_loss, total_train_accuracy, total_dev_accuracy, fake_img)

        # Plot Accuracy Graph
        train_acc_list.append(total_train_accuracy)
        dev_acc_list.append(total_dev_accuracy)
        utils.plot_graph(epoch, train_acc_list, dev_acc_list, 'Accuracy')

# In[ ]:

def testing_module(eval_loader):
    
    # Load the best model
    BEST_MODEL = '32_lsgan.tar'
    if os.path.isfile(BEST_MODEL):
        print("=> loading dis checkpoint")
        checkpoint = torch.load(BEST_MODEL)
        discriminator.load_state_dict(checkpoint['dis_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        generator.load_state_dict(checkpoint['gen_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(BEST_MODEL, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(BEST_MODEL))
    
    total_dev_accuracy = eval_module(eval_loader)
    return total_dev_accuracy


# In[ ]:

if args.mode == 'train':
    # Train a model and save the best one
    main_module()
else:
    # Test model performance on Dev/Test data
    final_accuracy = testing_module(dev_loader)
    print ("Accuracy on the Dev/Test data is = %f" %(final_accuracy))





# In[ ]:




