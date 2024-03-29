"""

Contains all used datasets which should be cached as it is done by PyTorch.
Path to cache must be specified.
<...>_ADV datasets are augmented datasets by Vanilla Adversarial Training.
Return tokenized documents.
WLADL for BERT is added here.

"""

from torch.utils.data import Dataset
import pandas as pd
import io
import os
from torchtext.data.utils import get_tokenizer
from WLADL import mask_replace_with_syns_add_noise

BASIC_PATH = '/cluster/scratch/herdem'  # root path to datasets


class AG_NEWS(Dataset):
    def __init__(self, tokenizer, model, split, with_defense=False, thesaurus=None, embedding=None):
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
        self.with_defense = with_defense
        if with_defense and model == 'BERT':
            self.temp_tokenizer = get_tokenizer('basic_english')
            self.thesaurus = thesaurus
            self.embedding = embedding

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, idx):
        if self.model == 'BERT':
            if self.with_defense:
                # trick for WLADL: tokenize using simple tokenizer, apply WLADL, join again, then apply BERT tokenizer
                return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join(mask_replace_with_syns_add_noise(self.temp_tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str))), self.thesaurus, self.embedding, self.model)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
            else:
                return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
        else:
            return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)))


class AG_NEWS_ADV(Dataset):
    def __init__(self, tokenizer, model, split, with_defense=False, thesaurus=None, embedding=None):
        super().__init__()
        self.num_classes = 4
        self.tokenizer = tokenizer
        self.model = model
        self.split = split
        if split == 'train':
            self.dataset = pd.read_csv(
                BASIC_PATH + '/AG_NEWS/AG_NEWS_ADV.csv')
        elif split == 'test':
            self.dataset = pd.read_csv(
                BASIC_PATH + '/AG_NEWS/test.csv', header=None)
        else:
            raise ValueError()
        self.with_defense = with_defense
        if with_defense and model == 'BERT':
            self.temp_tokenizer = get_tokenizer('basic_english')
            self.thesaurus = thesaurus
            self.embedding = embedding

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, idx):
        if self.split == 'train':
            label = self.dataset["label"][idx]
            text = self.dataset["txt"][idx]

            if self.model == 'BERT':
                return label, self.tokenizer(text, padding="max_length", return_tensors='pt', max_length=512, truncation=True)
            else:
                return label, self.tokenizer(text)
        else:
            if self.model == 'BERT':
                if self.with_defense:
                    # trick for WLADL: tokenize using simple tokenizer, apply WLADL, join again, then apply BERT tokenizer
                    return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join(mask_replace_with_syns_add_noise(self.temp_tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str))), self.thesaurus, self.embedding, self.model)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
                else:
                    return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
            else:
                return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)))


class IMDB(Dataset):
    def __init__(self, tokenizer, model, split, with_defense=False, thesaurus=None, embedding=None):
        super().__init__()
        self.num_classes = 2
        self.tokenizer = tokenizer
        self.model = model
        self.with_defense = with_defense
        if with_defense and model == 'BERT':
            self.temp_tokenizer = get_tokenizer('basic_english')
            self.thesaurus = thesaurus
            self.embedding = embedding

        def yield_data(label, files, path):
            for file in files:
                with io.open(path + file, encoding='utf8') as f:
                    yield label, f.read()
        # fetch all files
        if split == 'train':
            filelist_pos = os.listdir(
                BASIC_PATH + '/IMDB/aclImdb/' + split + '/pos')
            filelist_neg = os.listdir(
                BASIC_PATH + '/IMDB/aclImdb/' + split + '/neg')
        elif split == 'test':
            filelist_pos = os.listdir(
                BASIC_PATH + '/IMDB/aclImdb/' + split + '/pos')
            filelist_neg = os.listdir(
                BASIC_PATH + '/IMDB/aclImdb/' + split + '/neg')
        else:
            raise ValueError()
        df_pos = pd.DataFrame(yield_data(1, filelist_pos, BASIC_PATH + '/IMDB/aclImdb/' + split + '/pos/'),
                              columns=['label', 'data'])
        df_neg = pd.DataFrame(yield_data(0, filelist_neg, BASIC_PATH + '/IMDB/aclImdb/' + split + '/neg/'),
                              columns=['label', 'data'])
        self.dataset = pd.concat([df_pos, df_neg])

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, idx):
        if self.model == 'BERT':
            if self.with_defense:
                # trick for WLADL: tokenize using simple tokenizer, apply WLADL, join again, then apply BERT tokenizer
                return int(self.dataset.iloc[idx, 0]), self.tokenizer(' '.join(mask_replace_with_syns_add_noise(self.temp_tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str))), self.thesaurus, self.embedding, self.model)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
            else:
                return int(self.dataset.iloc[idx, 0]), self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
        else:
            return int(self.dataset.iloc[idx, 0]), self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)))


class IMDB_ADV(Dataset):
    def __init__(self, tokenizer, model, split, with_defense=False, thesaurus=None, embedding=None):
        super().__init__()
        self.num_classes = 2
        self.tokenizer = tokenizer
        self.model = model
        self.with_defense = with_defense
        self.split = split
        if with_defense and model == 'BERT':
            self.temp_tokenizer = get_tokenizer('basic_english')
            self.thesaurus = thesaurus
            self.embedding = embedding

        def yield_data(label, files, path):
            for file in files:
                with io.open(path + file, encoding='utf8') as f:
                    yield label, f.read()
        # fetch all files
        if split == 'train':
            self.dataset = pd.read_csv(
                BASIC_PATH + '/IMDB/IMDB_ADV.csv')
        elif split == 'test':
            filelist_pos = os.listdir(
                BASIC_PATH + '/IMDB/aclImdb/' + split + '/pos')
            filelist_neg = os.listdir(
                BASIC_PATH + '/IMDB/aclImdb/' + split + '/neg')
            df_pos = pd.DataFrame(yield_data(1, filelist_pos, BASIC_PATH + '/IMDB/aclImdb/' + split + '/pos/'),
                                  columns=['label', 'data'])
            df_neg = pd.DataFrame(yield_data(0, filelist_neg, BASIC_PATH + '/IMDB/aclImdb/' + split + '/neg/'),
                                  columns=['label', 'data'])
            self.dataset = pd.concat([df_pos, df_neg])
        else:
            raise ValueError()

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, idx):
        if self.split == 'train':
            label = self.dataset["label"][idx]
            text = self.dataset["txt"][idx]

            if self.model == 'BERT':
                return label, self.tokenizer(text, padding="max_length", return_tensors='pt', max_length=512, truncation=True)
            else:
                return label, self.tokenizer(text)
        else:
            if self.model == 'BERT':
                if self.with_defense:
                    # trick for WLADL: tokenize using simple tokenizer, apply WLADL, join again, then apply BERT tokenizer
                    return int(self.dataset.iloc[idx, 0]), self.tokenizer(' '.join(mask_replace_with_syns_add_noise(self.temp_tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str))), self.thesaurus, self.embedding, self.model)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
                else:
                    return int(self.dataset.iloc[idx, 0]), self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
            else:
                return int(self.dataset.iloc[idx, 0]), self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)))


class YahooAnswers(Dataset):
    def __init__(self, tokenizer, model, split, with_defense=False, thesaurus=None, embedding=None):
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
        self.with_defense = with_defense
        if with_defense and model == 'BERT':
            self.temp_tokenizer = get_tokenizer('basic_english')
            self.thesaurus = thesaurus
            self.embedding = embedding

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, idx):
        if self.model == 'BERT':
            if self.with_defense:
                # trick for WLADL: tokenize using simple tokenizer, apply WLADL, join again, then apply BERT tokenizer
                return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join(mask_replace_with_syns_add_noise(self.temp_tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str))), self.thesaurus, self.embedding, self.model)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
            else:
                return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
        else:
            return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)))


class YahooAnswers_ADV(Dataset):
    def __init__(self, tokenizer, model, split, with_defense=False, thesaurus=None, embedding=None):
        super().__init__()
        self.num_classes = 10
        if split == 'train':
            self.dataset = pd.read_csv(
                BASIC_PATH + '/YahooAnswers/YahooAnswers_ADV.csv')
        elif split == 'test':
            self.dataset = pd.read_csv(
                BASIC_PATH + '/YahooAnswers/yahoo_answers_csv/test.csv', header=None)
        else:
            raise ValueError()
        self.split = split
        self.tokenizer = tokenizer
        self.model = model
        self.with_defense = with_defense
        if with_defense and model == 'BERT':
            self.temp_tokenizer = get_tokenizer('basic_english')
            self.thesaurus = thesaurus
            self.embedding = embedding

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.split == 'train':
            label = self.dataset["label"][idx]
            text = self.dataset["txt"][idx]

            if self.model == 'BERT':
                return label, self.tokenizer(text, padding="max_length", return_tensors='pt', max_length=512, truncation=True)
            else:
                return label, self.tokenizer(text)
        else:
            if self.model == 'BERT':
                if self.with_defense:
                    # trick for WLADL: tokenize using simple tokenizer, apply WLADL, join again, then apply BERT tokenizer
                    return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join(mask_replace_with_syns_add_noise(self.temp_tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str))), self.thesaurus, self.embedding, self.model)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
                else:
                    return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
            else:
                return int(self.dataset.iloc[idx, 0]) - 1, self.tokenizer(' '.join((self.dataset.iloc[idx, 1:]).map(str)))
