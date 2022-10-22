"""The module containing the CNN model class. """
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """A class of CNN model, it contains three convolutional layers, ReLU activations, dropout layers for training,
    and two linear layers ended with softmax. """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(10 * 10 * 64, 256)  # TODO: parametrize linear function later
        self.fc2 = nn.Linear(256, 36)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.25, training=self.training)
        x = x.view(-1, 10 * 10 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
