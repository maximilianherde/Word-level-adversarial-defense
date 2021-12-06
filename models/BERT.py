from pytorch_pretrained_bert import BertModel
import torch.nn as nn


class BERTClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super(BERTClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x, masks=None):
        _, x = self.bert(x, attention_mask=masks, output_all_encoded_layers=False)
        x = self.dropout(x)
        x = self.linear(x)
        y = self.softmax(x)
        return y