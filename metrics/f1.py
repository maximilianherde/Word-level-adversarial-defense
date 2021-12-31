"""

F1 metric.
Uses sklearn to compute this metric on the CPU.

"""

import torch
from sklearn.metrics import f1_score


def f1(model, MODEL, data_loader, avg):
    """
    Returns the f1 score of the model (specified as MODEL) over data_loader. avg is the averging method used for multi-class problems.
    """
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        if MODEL == "BERT":
            for idx, (labels, input_ids, token_type_ids, attention_mask) in enumerate(data_loader):
                y_pred.append(model(input_ids, token_type_ids,
                              attention_mask).argmax(1))
                y_true.append(labels)
        else:
            for idx, (labels, text) in enumerate(data_loader):
                y_pred.append(model(text).argmax(1))
                y_true.append(labels)

    y_pred_t = torch.hstack(y_pred).to("cpu").numpy()
    y_true_t = torch.hstack(y_true).to("cpu").numpy()
    return f1_score(y_true_t, y_pred_t, average=avg)
