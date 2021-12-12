import torch
from sklearn.metrics import roc_auc_score, f1_score


def stats(model, MODEL, data_loader, avg):
    model.eval()
    y_true = []
    y_pred = []
    y_pred_arg = []
    with torch.no_grad():
        if MODEL == "BERT":
            for idx, (labels, input_ids, token_type_ids, attention_mask) in enumerate(data_loader):
                y_pred.append(model(input_ids, token_type_ids, attention_mask))
                y_pred_arg.append(model(input_ids, token_type_ids, attention_mask).argmax(1))
                y_true.append(labels)
        else:
            for idx, (labels, text) in enumerate(data_loader):
                y_pred.append(model(text))
                y_pred_arg.append(model(text).argmax(1))
                y_true.append(labels)

    y_pred_t = torch.vstack(y_pred).to("cpu").numpy()
    y_true_t = torch.hstack(y_true).to("cpu").numpy()
    y_pred_arg_t = torch.hstack(y_pred_arg).to("cpu").numpy()
    # for binary case only pass one prob. column
    if y_pred_t.shape[1] < 3:
        y_pred_t = y_pred_t[:, 1]

    acc = (y_pred_arg_t == y_true_t).sum().item()/len(y_true_t)
    roc_auc = roc_auc_score(y_true_t, y_pred_t, multi_class='ovr', average=avg)
    f1 = f1_score(y_true_t, y_pred_arg_t, average=avg)
    return acc, roc_auc, f1
