from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, dataset, num_classes, tokenizer):
        self.num_classes = num_classes
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        label, text = self.dataset.__getitem__(idx)
        if self.num_classes == 2:
            label_int = 1 if label == 'pos' else 0
            return label_int, self.tokenizer(text)
        else:
            return int(label) - 1, self.tokenizer(text)
