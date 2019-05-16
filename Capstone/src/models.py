import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.utils import weight_norm
import pdb


def initializer(m):
    # Run xavier on all weights and zero all biases
    if hasattr(m, 'weight'):
        if m.weight.ndimension() > 1:
            xavier_uniform_(m.weight.data)

    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.zero_() 


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


class DiscriminatorNet(torch.nn.Module):
    def __init__(self, args):
        super(DiscriminatorNet, self).__init__()
          
        dropout_rate = 0.45
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
    def __init__(self, args):
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


# For 64x64 image data
class DiscriminatorNet_64(torch.nn.Module):
    def __init__(self, args):
        super(DiscriminatorNet_64, self).__init__()
          
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
            weight_norm(nn.Conv2d(in_channels=filter1, out_channels=filter1, kernel_size=3, stride=1, padding=1), name='weight'),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(in_channels=filter1, out_channels=filter1, kernel_size=3, stride=1, padding=1), name='weight'),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=3, stride=2, padding=1), name='weight'),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate)
        )

        # CNNBlock 3
        self.wn_conv3 = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=3, stride=1, padding=1), name='weight'),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=3, stride=1, padding=1), name='weight'),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(in_channels=filter2, out_channels=filter2, kernel_size=3, stride=2, padding=1), name='weight'),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout_rate)
        )        
        
        # CNNBlock 4
        self.wn_conv4 = nn.Sequential(
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
        x = self.wn_conv4(x)
        
        # Linear
        flatten = x.mean(dim=3).mean(dim=2)
        linear = self.wn_linear(flatten)
        prob = self.softmax(linear)
        return flatten, linear, prob


# In[8]:


class GeneratorNet_64(torch.nn.Module):
    def __init__(self, args):
        super(GeneratorNet_64, self).__init__()
        
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
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.wn_deconv4 = nn.Sequential(
            weight_norm(nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1),
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
        x = self.deconv3(x)
        x = self.wn_deconv4(x)

        return x
