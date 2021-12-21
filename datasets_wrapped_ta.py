from datasets_euler import AG_NEWS, IMDB, YahooAnswers
import textattack


def get_textattack_AG_NEWS():
    ag_news = AG_NEWS(None, None, split='test')
    dataset = ag_news.dataset
    pass


def get_textattack_IMDB():
    imdb = IMDB(None, None, split='test')
    dataset = imdb.dataset
    pass


def get_textattack_YahooAnswers():
    yahoo = YahooAnswers(None, None, split='test')
    dataset = yahoo.dataset
    pass
