import torch
import torch.nn.functional as F
from torch.nn import Linear, Softmax, Conv2d, Dropout

class CNNClassifier(torch.nn.Module):
    def __init__(self, num_classes, in_channels, out_channels, kernel_heights, pad=0, stri=1, embed_dim=50, drop=0.2):
        super().__init__()
        self.conv1 = Conv2d(in_channels, out_channels[0], kernel_size=(kernel_heights[0], embed_dim), stride=stri, padding=pad)
        self.conv2 = Conv2d(in_channels, out_channels[1], kernel_size=(kernel_heights[1], embed_dim), stride=stri, padding=pad)
        self.conv3 = Conv2d(in_channels, out_channels[2], kernel_size=(kernel_heights[2], embed_dim), stride=stri, padding=pad)
        self.drop = Dropout(drop)
        self.fc = Linear(sum(out_channels), num_classes)
        self.soft = Softmax(dim=1)

    def _conv_n_maxpool_1d(self, input, conv_layer):

        conved = conv_layer(input) # conved.size() = (batch_size, out_channels[0], dim, 1)
        reld = F.relu(conved.squeeze(3)) # reld.size() = (batch_size, out_channels[0], dim)
        max_out = F.max_pool1d(reld, reld.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels[0])

        return max_out

    def forward(self, x):
        # x.size() = (batch_size, num_seq, embed_dim)
        x = x.unsqueeze(1) # x.size() = (batch_size, 1, num_seq, embed_dim)

        out_1 = self._conv_n_maxpool_1d(x, self.conv1)
        out_2 = self._conv_n_maxpool_1d(x, self.conv2)
        out_3 = self._conv_n_maxpool_1d(x, self.conv3)

        cat_out = torch.cat((out_1, out_2, out_3), dim=1)

        drop = self.drop(cat_out)
        fc_out = self.fc(drop)
        out = self.soft(fc_out)

        return out
