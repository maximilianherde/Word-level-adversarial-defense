from datasets_euler import AG_NEWS, IMDB, YahooAnswers
import textattack


def get_textattack_AG_NEWS():
    ag_news = AG_NEWS(None, None, split='test')
    dataset = ag_news.dataset
    pass


def get_textattack_IMDB():
    imdb = IMDB(None, None, split='test')
    dataset = imdb.dataset
    new_order_columns = ['data', 'label']
    new_dataset = dataset.reindex(columns=new_order_columns)
    return textattack.datasets.Dataset(new_dataset.values.tolist())


def get_textattack_YahooAnswers():
    yahoo = YahooAnswers(None, None, split='test')
    dataset = yahoo.dataset
    pass
