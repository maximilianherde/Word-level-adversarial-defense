"""

Contains the Convolutional Neural Networks for document classification.

"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Softmax, Conv2d, Dropout
import torch.nn as nn


class CNNClassifier(torch.nn.Module):
    """
    A CNN classifier using 2D-convolutions.
    Calculates class probabilities.
    """

    def __init__(self, num_classes, in_channels, out_channels, kernel_heights, pad=0, stri=1, embed_dim=50, drop=0.2):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels[0], kernel_size=(
            kernel_heights[0], embed_dim), stride=stri, padding=pad)
        self.conv2 = Conv2d(in_channels, out_channels[1], kernel_size=(
            kernel_heights[1], embed_dim), stride=stri, padding=pad)
        self.conv3 = Conv2d(in_channels, out_channels[2], kernel_size=(
            kernel_heights[2], embed_dim), stride=stri, padding=pad)
        self.drop = Dropout(drop)
        self.fc = Linear(sum(out_channels), num_classes)
        self.soft = Softmax(dim=1)

    def _conv_n_maxpool_1d(self, input, conv_layer):

        # conved.size() = (batch_size, out_channels[0], dim, 1)
        conved = conv_layer(input)
        # reld.size() = (batch_size, out_channels[0], dim)
        reld = F.relu(conved.squeeze(3))
        # maxpool_out.size() = (batch_size, out_channels[0])
        max_out = F.max_pool1d(reld, reld.size()[2]).squeeze(2)

        return max_out

    def forward(self, x):
        # x.size() = (batch_size, num_seq, embed_dim)
        x = x.unsqueeze(1)  # x.size() = (batch_size, 1, num_seq, embed_dim)

        out_1 = self._conv_n_maxpool_1d(x, self.conv1)
        out_2 = self._conv_n_maxpool_1d(x, self.conv2)
        out_3 = self._conv_n_maxpool_1d(x, self.conv3)

        cat_out = torch.cat((out_1, out_2, out_3), dim=1)

        drop = self.drop(cat_out)
        fc_out = self.fc(drop)
        out = self.soft(fc_out)

        return out


class CNNClassifier2(torch.nn.Module):
    """
    A CNN classifier using 1D-convolutions.
    Calculates class probabilites.
    """

    def __init__(self, num_classes, embedding_dim=50):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=28, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=28, out_channels=128, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU()
        )

        self.dense = nn.Sequential(
            # nn.Dropout(.3),
            nn.Linear(128, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes),
            nn.Softmax(dim=1)
        )

    def pooling(self, x):
        x_max = F.max_pool1d(x, x.size()[2]).squeeze(2)
        return x_max

    def forward(self, x):
        bs = x.shape[0]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pooling(x)
        x = x.view(bs, -1)
        y = self.dense(x)
        return y
