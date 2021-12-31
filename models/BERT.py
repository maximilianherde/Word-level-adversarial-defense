"""

A BERT sequence classifier.

"""


import torch.nn as nn
from transformers import BertForSequenceClassification

##########################################################################
# helper functions for fine-tuning last two layers of encoder and pooling
##########################################################################


def get_child(model, *arg):
    res = model
    for i in arg:
        res = list(res.children())[i]
    return res


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def custom_freezer(model):
    unfreeze_model(model)

    # freeze whole BertLayer
    for c in model.children():
        if str(c).startswith('Bert'):
            freeze_model(c)

    # unfreeze top 2 layer in BertEncoder
    bert_encoder = get_child(model, 0, 1, 0)
    for i in range(1, 3):
        m = bert_encoder[-i]
        unfreeze_model(m)

    # unfreeze Pooling layer
    bert_pooling = get_child(model, 0, 2)
    unfreeze_model(bert_pooling)

    print('Trainable parameters: {}'.format(count_parameters(model, True)))
    return model

##########################################################################


class BERTClassifier(nn.Module):
    """
    A BERT sequence classifier.
    The last two layers of the encoder as well as the pooling layer can be fine-tuned/trained.
    Approx 15 million parameters.
    Calculates class probabilites.
    """

    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()

        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=num_classes)
        self.bert = custom_freezer(self.bert)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        y = self.bert(input_ids, token_type_ids, attention_mask)
        return self.softmax(y.logits)
