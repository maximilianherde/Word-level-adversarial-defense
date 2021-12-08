from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, dataset, num_classes, tokenizer, model):
        self.num_classes = num_classes
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        label, text = self.dataset.__getitem__(idx)
        if type(label) == str:
            if label == 'neg':
                label = 0
            else:
                label = 1
        else:
            label = int(label) - 1

        if self.model == 'BERT':
            return label, self.tokenizer(text, padding="max_length", return_tensors='pt', max_length=512, truncation=True)
        else:
            return label, self.tokenizer(text)
