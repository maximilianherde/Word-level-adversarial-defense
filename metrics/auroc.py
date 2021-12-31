"""

Area under Receiver-operator characteristics metric.
Uses sklearn to compute this metric on the CPU.

"""

import torch
from sklearn.metrics import roc_auc_score


def auroc(model, MODEL, data_loader, avg):
    """
    Returns the auroc metric for data in data_loader on the model (specified as MODEL) using avg as average for multi-class problems.
    """
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        if MODEL == "BERT":
            for idx, (labels, input_ids, token_type_ids, attention_mask) in enumerate(data_loader):
                y_pred.append(model(input_ids, token_type_ids, attention_mask))
                y_true.append(labels)
        else:
            for idx, (labels, text) in enumerate(data_loader):
                y_pred.append(model(text))
                y_true.append(labels)

    y_pred_t = torch.vstack(y_pred).to("cpu").numpy()
    y_true_t = torch.hstack(y_true).to("cpu").numpy()
    # for binary case only pass one prob. column
    if y_pred_t.shape[1] < 3:
        y_pred_t = y_pred_t[:, 1]

    return roc_auc_score(y_true_t, y_pred_t, multi_class='ovr', average=avg)
