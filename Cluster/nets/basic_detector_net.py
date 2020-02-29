import torch.nn as nn
import torch.nn.functional as F

class BasicDetectorNetwork(nn.Module):
    def __init__(self, input_shape, dim_reduction_type, num_filters, num_layers, use_bias=False):
        """
        Initializes a convolutional network module object.
        :param input_shape: The shape of the inputs going in to the network.
        :param dim_reduction_type: The type of dimensionality reduction to apply after each convolutional stage, should be one of ['max_pooling', 'avg_pooling', 'strided_convolution', 'dilated_convolution']
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every conv layer, except dim reduction stages, where those are automatically infered.
        :param num_layers: Number of conv layers (excluding dim reduction stages)
        :param use_bias: Whether our convolutions will use a bias.
        """
        super(BasicDetectorNetwork, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.dim_reduction_type = dim_reduction_type
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # obejctive is to bring down the image size to single unit-->
        # here given image size is 224x224px
        self.conv1 = nn.Conv2d(1, 32, 5)
        # 224--> 224-5+1=220
        self.pool1 = nn.MaxPool2d(2, 2)
        # 220/2=110 ...(32,110,110)

        self.conv2 = nn.Conv2d(32, 64, 3)
        # 110--> 110-3+1=108
        self.pool2 = nn.MaxPool2d(2, 2)
        # 108/2=54

        self.conv3 = nn.Conv2d(64, 128, 3)
        # 54-->54-3+1=52
        self.pool3 = nn.MaxPool2d(2, 2)
        # 52/2=26

        self.conv4 = nn.Conv2d(128, 256, 3)
        # 26-->26-3+1=24
        self.pool4 = nn.MaxPool2d(2, 2)
        # 24/2=12

        self.conv5 = nn.Conv2d(256, 512, 1)
        # 12-->12-1+1=12
        self.pool5 = nn.MaxPool2d(2, 2)
        # 12/2=6

        # 6x6x512
        self.fc1 = nn.Linear(6 * 6 * 512, 1024)
        #         self.fc2 = nn.Linear(1024,1024)
        self.fc2 = nn.Linear(1024, 136)

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p = 0.6)
        # self.fc2_drop = nn.Dropout(p=.5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))
        x = self.drop5(self.pool5(F.relu(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = self.drop6(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        return
