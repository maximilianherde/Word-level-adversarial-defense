from datasets_euler import AG_NEWS, IMDB, YahooAnswers
import pandas as pd
import textattack


def label_map(x):
    return int(x) - 1


def concat_text(x):
    return ' '.join(x[:].map(str))


def get_textattack_AG_NEWS():
    ag_news = AG_NEWS(None, None, split='test')
    dataset = ag_news.dataset
    print(dataset.head())
    new_dataset = pd.DataFrame(columns=['data', 'label'])
    new_dataset['label'] = dataset.iloc[:, 0].apply(label_map)
    new_dataset['data'] = dataset.iloc[:, 1:].apply(concat_text, axis=1)
    print(new_dataset.head())
    return textattack.datasets.Dataset(new_dataset.values.tolist())


def get_textattack_IMDB():
    imdb = IMDB(None, None, split='test')
    dataset = imdb.dataset
    new_order_columns = ['data', 'label']
    new_dataset = dataset.reindex(columns=new_order_columns)
    return textattack.datasets.Dataset(new_dataset.values.tolist())


def get_textattack_YahooAnswers():
    yahoo = YahooAnswers(None, None, split='test')
    dataset = yahoo.dataset
    new_dataset = pd.DataFrame(columns=['data', 'label'])
    new_dataset['label'] = dataset.iloc[:, 0].apply(label_map)
    new_dataset['data'] = dataset.iloc[:, 1:].apply(concat_text, axis=1)
    return textattack.datasets.Dataset(new_dataset.values.tolist())
