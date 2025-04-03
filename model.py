from torch import nn
import torch.nn.functional as F

class BadNet(nn.Module):
    """BadNet model with dynamic input handling for MNIST, CIFAR-10, and other datasets."""
    def __init__(self, input_size=3, output=10, img_dim=32):
        """
        Args:
            input_size (int): Number of input channels (e.g., 3 for RGB, 1 for grayscale).
            output (int): Number of output classes.
            img_dim (int): Dimension of the input images (assumes square images).
        """
        super().__init__()
        self.input_size = input_size
        self.output = output
        self.img_dim = img_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Dynamically compute flattened feature size
        conv_out_dim = self._calculate_conv_output_dim(img_dim)
        self.fc_features = conv_out_dim * conv_out_dim * 32  # 32 is the number of filters in conv2

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_features, 512)
        self.fc2 = nn.Linear(512, output)

    def _calculate_conv_output_dim(self, dim):
        """
        Helper function to calculate output dimensions after convolution and pooling.
        Args:
            dim (int): Dimension of the square input image.
        Returns:
            int: Dimension of the output square feature map.
        """
        conv1_out = dim - 4  # Convolution with kernel_size=5 reduces size by 4
        pool1_out = conv1_out // 2  # Pooling reduces size by half
        conv2_out = pool1_out - 4
        pool2_out = conv2_out // 2
        return pool2_out

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        if latent:
            return None, x  # skip final classification, return penultimate features
        x = self.fc2(x)
        return x

