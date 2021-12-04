import torch
from torch.nn import LSTM, GRU, Linear, Softmax


class BidirectionalLSTMClassifier(torch.nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.LSTM = LSTM(50, hidden_size, num_layers=num_layers,
                         batch_first=True, bidirectional=True)
        self.linear = Linear(2 * hidden_size, num_classes)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        _, (h_n, _) = self.LSTM(x)
        h_forward = h_n[2 * self.num_layers - 2]
        h_backward = h_n[2 * self.num_layers - 1]
        y = self.linear(torch.cat((h_forward, h_backward), 1))
        return self.softmax(y)


class BidirectionalGRUClassifier(torch.nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.GRU = GRU(50, hidden_size, num_layers=num_layers,
                       batch_first=True, bidirectional=True)
        self.linear = Linear(2 * hidden_size, num_classes)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        _, h_n = self.GRU(x)
        h_forward = h_n[2 * self.num_layers - 2]
        h_backward = h_n[2 * self.num_layers - 1]
        y = self.linear(torch.cat((h_forward, h_backward), 1))
        return self.softmax(y)
