"""
    Code to generate adversarial examples to augment for VAT training.
    Uses a LSTM_X_CLEAN.pt model and PWWS attack, logs the adversarial
    samples, corresponding to approximately 10% of each dataset.
"""


import textattack
import sys
import torch
from models.BiRLM import BidirectionalGRUClassifier, BidirectionalLSTMClassifier
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
from attackutils.modelwrapper import CustomPyTorchModelWrapper
from datasets_euler import AG_NEWS, IMDB, YahooAnswers
import pandas as pd

def label_map(x):
    return int(x) - 1


def concat_text(x):
    return ' '.join(x[:].map(str))


def get_textattack_AG_NEWS(split='train'):
    ag_news = AG_NEWS(None, None, split=split)
    dataset = ag_news.dataset
    new_dataset = pd.DataFrame(columns=['data', 'label'])
    new_dataset['label'] = dataset.iloc[:, 0].apply(label_map)
    new_dataset['data'] = dataset.iloc[:, 1:].apply(concat_text, axis=1)
    return textattack.datasets.Dataset(new_dataset.values.tolist())


def get_textattack_IMDB(split='train'):
    imdb = IMDB(None, None, split=split)
    dataset = imdb.dataset
    new_order_columns = ['data', 'label']
    new_dataset = dataset.reindex(columns=new_order_columns)
    return textattack.datasets.Dataset(new_dataset.values.tolist())


def get_textattack_YahooAnswers(split='train'):
    yahoo = YahooAnswers(None, None, split=split)
    dataset = yahoo.dataset
    new_dataset = pd.DataFrame(columns=['data', 'label'])
    new_dataset['label'] = dataset.iloc[:, 0].apply(label_map)
    new_dataset['data'] = dataset.iloc[:, 1:].apply(concat_text, axis=1)
    return textattack.datasets.Dataset(new_dataset.values.tolist())

if len(sys.argv) == 1:
    print(
        'Usage: python adv_dataset_generator.py DATASET GLOVE_CACHE_PATH TRANSFORMERS_CACHE_PATH MODEL_PATH CSV_PATH')
    print('Choices for DATASET: IMDB, AG_NEWS, YahooAnswers')
    exit()

else:
    DATASET = sys.argv[1]
    VECTOR_CACHE = sys.argv[2]
    TRANSFORMERS_CACHE = sys.argv[3]
    MODEL_PATH = sys.argv[4]
    CSV_PATH = sys.argv[5]

print(f'Generating Adv. Dataset with: {DATASET} {VECTOR_CACHE} {TRANSFORMERS_CACHE} {MODEL_PATH} {CSV_PATH}')

model_name = "LSTM" + '_' + DATASET + '_' + "CLEAN"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = get_tokenizer('basic_english')

if DATASET == "AG_NEWS":
    dataset = get_textattack_AG_NEWS()
    num_classes = 4
    len_examples = 12000
elif DATASET == "IMDB":
    dataset = get_textattack_IMDB()
    num_classes = 2
    len_examples = 2500
elif DATASET == "YahooAnswers":
    dataset = get_textattack_YahooAnswers()
    num_classes = 10
    len_examples = 140000
else:
    raise ValueError()

model = BidirectionalLSTMClassifier(num_classes, 64, 1).to(device)
checkpoint = torch.load(MODEL_PATH + '/' + model_name + '.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model_wrapper = CustomPyTorchModelWrapper(model, outdim=num_classes,
                                          vocab=GloVe(name='6B', dim=50, cache=VECTOR_CACHE), device=device)
attack = textattack.attack_recipes.pwws_ren_2019.PWWSRen2019.build(
        model_wrapper)
attack_args = textattack.AttackArgs(
        num_examples=len_examples, log_to_csv=CSV_PATH+'/'+DATASET+"_ADV.csv", csv_coloring_style="plain", checkpoint_interval=250)
attacker = textattack.Attacker(attack, dataset, attack_args)
attacker.attack_dataset()
