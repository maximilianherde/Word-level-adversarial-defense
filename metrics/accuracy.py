import torch


def accuracy(model, MODEL, data_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        if MODEL == "BERT":
            for idx, (labels, input_ids, token_type_ids, attention_mask) in enumerate(data_loader):
                y_pred.append(model(input_ids, token_type_ids, attention_mask).argmax(1))
                y_true.append(labels)
        else:
            for idx, (labels, text) in enumerate(data_loader):
                y_pred.append(model(text).argmax(1))
                y_true.append(labels)

    y_pred_t = torch.hstack(y_pred)
    y_true_t = torch.hstack(y_true)
    return (y_pred_t == y_true_t).sum().item()/len(y_true_t)
