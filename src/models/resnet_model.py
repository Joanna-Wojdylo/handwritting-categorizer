""" The module containing ResNet model. """

import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class MnistResNet(nn.Module):
    """A class containing a ResNet model. Used the pretrained ResNet model from torchvision.models and matched the
    input and output layers to the problem being solved. In addition, a softmax was added to the output layer to
    maintain the convention adopted in the project. """
    def __init__(self, in_channels=1):
        super(MnistResNet, self).__init__()

        # Load a pretrained resnet model from torchvision.models in Pytorch
        self.model = models.resnet18(pretrained=True)

        # Change the input layer to take Grayscale image, instead of RGB images.
        # Hence, in_channels is set as 1
        # original definition of the first layer on the ResNet class:
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Change the output layer to output 36 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 36)

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)
