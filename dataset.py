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
        return int(label) - 1, self.tokenizer(text)
