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
# from logger import Logger
from PIL import Image


graph_dir = 'result_graphs'


# Create Dataset
class TCGADataset(Dataset):
    def __init__(self, args, image_size, split):
        self.split = split
        self.args = args
        self.tcga_dataset = self._create_dataset(image_size, split)
        self.patches, self.labels = self.tcga_dataset
        self.label_mask = self._create_label_mask()
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        
    def balance_data(self, images, labels):
        cancer = np.count_nonzero(labels)
        noncancer = (labels.shape[0] - cancer)
        minimum = min(cancer, noncancer)
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
        print("Cancerous patches : ", len(sample_idxs_cancer))
        print("Non cancerous patches : ", len(sample_idxs_nocancer))
        print("-------------------------------------------------")
        
        return images, labels
    
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
        #images, labels = self.balance_data(images, labels)            
        return images, labels
        
    def _one_hot(self, y):
        label = y
        label_onehot = np.zeros(self.args.num_classes + 1)
        label_onehot[label] = 1
        return label_onehot
    
    def _create_label_mask(self):
        if self.split == 'train':
            l = len(self.labels)
            label_mask = np.zeros(l)
            masked_len = int(self.args.labeled_rate * l)
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


# Get dataloaders
def get_loader(args):
    #num_workers = 2

    tcga_train = TCGADataset(args, image_size=args.image_size, split='train')
    tcga_dev = TCGADataset(args, image_size=args.image_size, split='dev')
#     tcga_test = TCGADataset(image_size=image_size, split='test')

    train_loader = DataLoader(
        dataset=tcga_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
        #num_workers=num_workers
    )

    dev_loader = DataLoader(
        dataset=tcga_dev,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
        #num_workers=num_workers
    )
    
#     test_loader = DataLoader(
#         dataset=tcga_test,
#         batch_size=batch_size,
#         shuffle=True
#         #num_workers=num_workers
#     )

    return train_loader, dev_loader#, test_loader


def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): 
        return n.cuda() 
    return n


def save_checkpoint(state, args, is_best):
    torch.save(state, args.param + '.tar')
    if is_best:
        shutil.copyfile(args.param + '.tar', 'best_' + args.param + '.tar')


def tensorboard_logging(args, epoch, G_loss, D_loss, total_train_accuracy, total_dev_accuracy, fake_img):
    image_dir = 'images_' + args.param

    # 1. Log scalar values (scalar summary)
    info = { 'Epoch': epoch, 'G_loss': G_loss, 'D_loss': D_loss, 'train_accuracy': total_train_accuracy, 'dev_accuracy': total_dev_accuracy }
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
    info = { image_dir : fake_img.view(-1, args.image_size, args.image_size)[:10].detach().cpu().numpy() }

    for tag, images in info.items():
        logger.image_summary(tag, images, epoch)



def plot_graph(args, epoch, train, dev, mode):
    epoch_list = np.arange(epoch + 1)
    plt.plot(epoch_list, train)
    plt.plot(epoch_list, dev)
    
    if mode.lower() == 'accuracy':
        location = 'lower right'
    else:
        location = 'upper right'

    plt.legend(['Train ' +  mode, 'Dev ' + mode], loc=location)
    plt.xlabel('Epochs')
    plt_image_path = os.path.join(graph_dir, args.param + '_' + mode.lower()[:4] + '_epoch_' + str(epoch))
    plt.savefig(plt_image_path)
