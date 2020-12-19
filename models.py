## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# initialize weights
import torch.nn.init as I

# only square images of width = height = 224 pixel can be processed 
image_wh = 224
# outsize of trained model / cnn. double the size of facial keypoints (2 dimensions x, y)
keypoint_size = 136

# calculate the resulting dimension size of convolutional layer
def conv_size(input_size, kernel_size, stride, padding_size):
    conv_size = (input_size + 2*padding_size - kernel_size)//stride + 1
    return conv_size

# calculate the resulting dimension size after maxpooling
def pool_size(input_size, kernel_size, stride):
    pool_size = (input_size - kernel_size)//stride + 1
    return pool_size

# calculate the input size (output size after last maxpooling layer) for the fully connected layers
def linear_size(depth, input_size):
    linear_size = depth * input_size**2
    return linear_size

class Net22(nn.Module):
    
    # definition of cnn layer 1 (conv2d and maxpool) parameters
    cnn_1 = {
    'in_channel_size': 1,   # gray image
    'out_channel_size': 32, # number of filters in conv layer 1 
    'kernel_size': 4,       # filter / kernel size [px]
    'stride_conv': 2,       # stride of conv kernel
    'padding_size': 0,      # no padding
    'window_size_pool': 3,  # size of maxpool filter
    'stride_pool': 2        # stride of maxpool kernel
    }
    
    # definition of cnn layer 2 (conv2d and maxpool) parameters
    cnn_2 = {
    'in_channel_size': cnn_1['out_channel_size'],
    'out_channel_size': 64,
    'kernel_size': 3,
    'stride_conv': 1,
    'padding_size': 0,
    'window_size_pool': 2,
    'stride_pool': 2
    }   
    
    # feature size after first fully connected layer
    fc1_output_features = 1024
    
    def __init__(self):
        super().__init__()
        # setup convolutional2d layer
        self.conv1 = nn.Conv2d(self.cnn_1['in_channel_size'], self.cnn_1['out_channel_size'], self.cnn_1['kernel_size'], stride=self.cnn_1['stride_conv'])
        size_conv1 = conv_size(image_wh, self.cnn_1['kernel_size'], self.cnn_1['stride_conv'], self.cnn_1['padding_size'])
        
        # setup maxpool2d layer
        self.pool1 = nn.MaxPool2d(self.cnn_1['window_size_pool'], stride=self.cnn_1['stride_pool'])
        size_pool1 = pool_size(size_conv1, self.cnn_1['window_size_pool'], self.cnn_1['stride_pool'])

        # second layers
        self.conv2 = nn.Conv2d(self.cnn_2['in_channel_size'], self.cnn_2['out_channel_size'], self.cnn_2['kernel_size'], stride=self.cnn_2['stride_conv'])
        size_conv2 = conv_size(size_pool1, self.cnn_2['kernel_size'], self.cnn_2['stride_conv'], self.cnn_2['padding_size'])

        self.pool2 = nn.MaxPool2d(self.cnn_2['window_size_pool'], stride=self.cnn_2['stride_pool'])
        size_pool2 = pool_size(size_conv2, self.cnn_2['window_size_pool'], self.cnn_2['stride_pool'])        
        
        # setup fully connected layers
        self.size_linear_input1 = linear_size(self.cnn_2['out_channel_size'], size_pool2)
        self.fc1 = nn.Linear(self.size_linear_input1, self.fc1_output_features)
        
        self.size_linear_input2 = self.fc1_output_features
        self.fc2 = nn.Linear(self.size_linear_input2, keypoint_size)
    
    # forward pass. relu activation function after convolutional layer, then maxpool
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # flatten dimensions before relu activation function
        x = x.view(-1, self.size_linear_input1)
        # 2 fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Net22d(Net22):
    # define dropout probability 
    p = 0.4
    
    def __init__(self):
        super().__init__()
        # setup dropout layer
        self.drop1 = nn.Dropout(self.p)
    
    # basically same forward pass like in net22 model
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, self.size_linear_input1)
        x = F.relu(self.fc1(x))
        # additional dropout layer in fully connected network
        x = self.drop1(x)
        x = self.fc2(x)
        return x
        