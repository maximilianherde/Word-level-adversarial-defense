from torch.utils.data import Dataset
import pandas as pd

BASIC_PATH = '/cluster/scratch/herdem'


class AG_NEWS(Dataset):
    def __init__(self, tokenizer, model, split):
        super().__init__()
        self.num_classes = 4
        self.tokenizer = tokenizer
        self.model = model
        if split == 'train':
            self.dataset = pd.read_csv(
                BASIC_PATH + '/AG_NEWS/train.csv', header=None)
        elif split == 'test':
            self.dataset = pd.read_csv(
                BASIC_PATH + '/AG_NEWS/test.csv', header=None)
        else:
            raise ValueError()

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, idx):
        if self.model == 'BERT':
            return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
        else:
            return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)))


class IMDB(Dataset):
    def __init__(self, tokenizer, model, split):
        super().__init__()
        self.num_classes = 2
        self.tokenizer = tokenizer
        self.model = model

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class YahooAnswers(Dataset):
    def __init__(self, tokenizer, model, split):
        super().__init__()
        self.num_classes = 10
        self.tokenizer = tokenizer
        self.model = model
        if split == 'train':
            self.dataset = pd.read_csv(
                BASIC_PATH + '/YahooAnswers/yahoo_answers_csv/train.csv', header=None)
        elif split == 'test':
            self.dataset = pd.read_csv(
                BASIC_PATH + '/YahooAnswers/yahoo_answers_csv/test.csv', header=None)
        else:
            raise ValueError()

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, idx):
        if self.model == 'BERT':
            return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
        else:
            return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)))
