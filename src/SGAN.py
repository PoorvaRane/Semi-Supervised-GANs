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
from logger import Logger
from PIL import Image


# In[2]:


logger = Logger('./logs')

log_path = './SSL_TCGA_log.csv'
model_path ='./TCGA_32.tar'


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
parser.add_argument('--image_dir', type=str, default='tcga_images_32', help='directory to save images')
parser.add_argument('--model_path', type=str, default='_32.tar', help='directory to save images')
args = parser.parse_args()

# Print args
print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')


# In[4]:


logger = Logger('./logs')

os.makedirs(args.image_dir, exist_ok=True)
os.makedirs(args.image_dir + '_fixed', exist_ok=True)


# In[3]:


# Create Dataset
class TCGADataset(Dataset):
    def __init__(self, image_size, split):
        self.split = split
        self.tcga_dataset = self._create_dataset(image_size, split)
        self.patches, self.labels = self.tcga_dataset
        self.label_mask = self._create_label_mask()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
#         self.save_images()
        
    def _create_dataset(self, image_size, split):
        data_dir = '/mys3bucket/patch_data'
        data_dir = os.path.join(data_dir, self.split)
            
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
        return images, labels
    
    def save_images(self):
        images = self.patches
        folder = 'tcga_check/'
        os.makedirs(folder, exist_ok=True)
        for i in range(args.batch_size):
            image = images[i]
            im = Image.fromarray(image)
            im.save(folder + str(i) + '.jpg', format='JPEG')
        
    def _one_hot(self, y):
        label = y
        label_onehot = np.zeros(args.num_classes + 1)
        label_onehot[label] = 1
        return label_onehot
    
    def _create_label_mask(self):
        if self.split == 'train':
            l = len(self.labels)
            label_mask = np.zeros(l)
            masked_len = int(args.labeled_rate * l)
            label_mask[0:masked_len] = 1
            np.random.shuffle(label_mask)
            label_mask = torch.LongTensor(label_mask)
            if torch.cuda.is_available(): 
                label_mask = label_mask.cuda()
            return label_mask
        return None

    def __getitem__(self, idx):
        data, label = self.patches[idx], self.labels[idx]
        label_onehot = self._one_hot(label)
        if self.split == 'train':
            return self.transform(Image.fromarray(data)), label, label_onehot, self.label_mask[idx]
        return self.transform(Image.fromarray(data)), label

    def __len__(self):
        return len(self.labels)


# In[4]:


# Get dataloaders
def get_loader(image_size, batch_size):
    #num_workers = 2

    tcga_train = TCGADataset(image_size=image_size, split='train')
    tcga_dev = TCGADataset(image_size=image_size, split='dev')
#     tcga_test = TCGADataset(image_size=image_size, split='test')

    train_loader = DataLoader(
        dataset=tcga_train,
        batch_size=batch_size,
        shuffle=True
        #num_workers=num_workers
    )

    dev_loader = DataLoader(
        dataset=tcga_dev,
        batch_size=batch_size,
        shuffle=True
        #num_workers=num_workers
    )
    
#     test_loader = DataLoader(
#         dataset=tcga_test,
#         batch_size=batch_size,
#         shuffle=True
#         #num_workers=num_workers
#     )

    return train_loader, dev_loader#, test_loader


# In[5]:


def initializer(m):
    # Run xavier on all weights and zero all biases
    if hasattr(m, 'weight'):
        if m.weight.ndimension() > 1:
            xavier_uniform_(m.weight.data)

    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.zero_() 


# In[6]:


class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            noise = x.new(x.size()).normal_(std=self.sigma)
            return x + noise
        else:
            return x


# In[7]:


class DiscriminatorNet(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
          
        dropout_rate = 0.5
        filter1 = 96
        filter2 = 192
        
        self.begin = nn.Sequential(
            GaussianNoise(0.05),
            nn.Dropout2d(0.2)   
        )
        
        # Conv operations
        # CNNBlock 1
        self.wn_conv1 = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels=args.num_channels, out_channels=filter1, kernel_size=3, stride=1, padding=1), name='weight'),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(in_channels=filter1, out_channels=filter1, kernel_size=3, stride=1, padding=1), name='weight'),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(in_channels=filter1, out_channels=filter1, kernel_size=3, stride=2, padding=1), name='weight'),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate)
        )
        
        # CNNBlock 2
        self.wn_conv2 = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=3, stride=1, padding=1), name='weight'),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=3, stride=1, padding=1), name='weight'),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=3, stride=2, padding=1), name='weight'),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate)
        )
        
        # CNNBlock 3
        self.wn_conv3 = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=3, stride=1, padding=0), name='weight'),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=1, stride=1, padding=0), name='weight'),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=1, stride=1, padding=0), name='weight'),
            nn.LeakyReLU(0.2)
        )
                
        # Linear 
        self.wn_linear = weight_norm(nn.Linear(in_features=filter2, out_features=(args.num_classes + 1)), name='weight')
        self.softmax = nn.Softmax(dim=1)
        self.apply(initializer)
        
    def forward(self, x):
        x = self.begin(x)
        # Convolutional Operations
        x = self.wn_conv1(x)
        x = self.wn_conv2(x)
        x = self.wn_conv3(x)
        
        # Linear
        flatten = x.mean(dim=3).mean(dim=2)
        linear = self.wn_linear(flatten)
        prob = self.softmax(linear)
        return flatten, linear, prob


# In[8]:


class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=args.latent_size, out_features=4 * 4 * 512, bias=False),
            nn.BatchNorm1d(4 * 4 * 512),
            nn.ReLU()
            )
        # Conv operations
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.wn_deconv3 = nn.Sequential(
            weight_norm(nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1),
                        name='weight'),
            nn.Tanh()
        )
        self.apply(initializer)
        
    def forward(self, x):
        
        x = self.linear1(x)
        x = x.view(-1, 512, 4, 4)
        
        # Deconvolutional Operations
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.wn_deconv3(x)

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
train_loader, dev_loader = get_loader(args.image_size, args.batch_size)

# Optimizers
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lrD, betas=(args.b1, args.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lrG, betas=(args.b1, args.b2))

# Loss
cross_loss = nn.CrossEntropyLoss(reduction='none')

if torch.cuda.is_available():
    discriminator = discriminator.cuda()
    generator = generator.cuda()
    discriminator = nn.DataParallel(discriminator)
    generator = nn.DataParallel(generator)
    cross_loss = cross_loss.cuda()


# In[11]:


# Visualize Data
def plot_fake_data(data, grid_size = [5, 5]):
    _, axes = plt.subplots(figsize = grid_size, nrows = grid_size[0], ncols = grid_size[1],
                           sharey = True, sharex = True)

    size = grid_size[0] * grid_size[1]
    index = np.int_(np.random.uniform(0, data.shape[0], size = (size)))

    figs = data[index].reshape(-1, args.image_size, args.image_size)

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


# In[ ]:


def test_discriminator(b_size, img, label):
    
    # Generate Fake Image
    z = noise(b_size)
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


def train_generator(optimizer_G, b_size, epsilon):
    
    # Generate Fake Image
    z = noise(b_size)
    fake_img = generator(z)

    # Discriminator outputs for real and fake
    d_real_flatten, d_real_linear, d_real_prob = discriminator(img)
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


def save_checkpoint(state, is_best, model_type):
    torch.save(state, model_type + args.model_path)
    if is_best:
        shutil.copyfile(model_type + args.model_path, model_type + '_best_' +args.model_path)


# In[15]:


# Fixed noise vector
fixed_z = noise(args.batch_size)

for epoch in range(args.num_epochs):
    total_train_accuracy = 0
    total_dev_accuracy = 0
    G_loss = 0
    D_loss = 0
    
    # TRAIN LOADER
    generator.train()
    discriminator.train()
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
            g_loss, fake_img = train_generator(optimizer_G, b_size, args.epsilon)
            batch_g_loss += g_loss.item()
        batch_g_loss = batch_g_loss/float(args.generator_frequency)
       
        total_train_accuracy += train_batch_accuracy
        D_loss += batch_d_loss
        G_loss += batch_g_loss    
        
        if i%b_size == b_size-1:
            print("Train [Epoch %d/%d] [Batch %d/%d] [D loss: %f, train acc: %.3f%%] [G loss: %f]" % (epoch, args.num_epochs,
                          i, len(train_loader), d_loss.item(), 100 * train_batch_accuracy, g_loss.item()))

    # Epoch statistics
    total_train_accuracy = total_train_accuracy/float(i+1)
    avg_D_loss = D_loss/float(i+1)
    avg_G_loss = G_loss/float(i+1)
            
    # DEV LOADER
    generator.eval()
    discriminator.eval()
    for i, data in enumerate(dev_loader):
        
        img, label = data
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        
        b_size = img.size(0)
        dev_accuracy = test_discriminator(b_size, img, label)
        total_dev_accuracy += d_accuracy
        
        if i%b_size == b_size-1:
            print("Dev [Epoch %d/%d] [Batch %d/%d] [D acc: %.3f%%]" % (epoch, args.num_epochs,
                          i, len(dev_loader), 100 * dev_accuracy))
        
    # Print Epoch results
    total_dev_accuracy = total_dev_accuracy/float(i+1)

    is_best = total_dev_accuracy >= total_train_accuracy
    # Save best model
    save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': discriminator.state_dict(),
    'optimizer' : optimizer_D.state_dict(),
    }, is_best, 'dis')
    save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': generator.state_dict(),
    'optimizer' : optimizer_G.state_dict(),
    }, is_best, 'gen')
    
    print('--------------------------------------------------------------------')
    print("===> [Epoch %d/%d] [Avg D loss: %f, avg train acc: %.3f%%, avg dev acc: %.3f%%] [Avg G loss: %f]" % (epoch, args.num_epochs, 
                                                  avg_D_loss, 100 * total_train_accuracy, 100* total_dev_accuracy, avg_G_loss))
    print('--------------------------------------------------------------------')
    
    # Save Images
    save_image(fake_img, args.image_dir + '/epoch_%d_batch_%d.png' % (epoch, i), nrow=8, normalize=True)
    # Save Fixed Images
    fixed_fake_img = generator(fixed_z)
    save_image(fixed_fake_img, args.image_dir + '_fixed' + '/epoch_%d_batch_%d.png' % (epoch, i), nrow=8, normalize=True)
    
    # Tensorboard logging 
    
    # 1. Log scalar values (scalar summary)
    info = { 'Epoch': epoch, 'G_loss': avg_G_loss, 'D_loss': avg_D_loss, 'train_accuracy': total_train_accuracy, 'dev_accuracy': total_dev_accuracy }
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)
    
    # 2. Log values and gradients of the parameters (histogram summary)
    # Generator summary
    for tag, value in generator.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.detach().cpu().numpy(), epoch)
        logger.histo_summary(tag+'/grad', value.grad.detach().cpu().numpy(), epoch)
    
    #Discriminator summary
    for tag, value in discriminator.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.detach().cpu().numpy(), epoch)
        logger.histo_summary(tag+'/grad', value.grad.detach().cpu().numpy(), epoch)
        
    # 3. Log generated images (image summary)
    info = { args.image_dir : fake_img.view(-1, args.image_size, args.image_size)[:10].detach().cpu().numpy() }

    for tag, images in info.items():
        logger.image_summary(tag, images, epoch)


# In[ ]:





# In[ ]:





# In[ ]:




