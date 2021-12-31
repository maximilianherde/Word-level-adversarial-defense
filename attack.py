"""

Script for attacking all models using BAE-R, PWWS, and Faster Genetic Algorithm.
It is thought to be used in a firewalled environment such as the compute nodes of ETHZ's Euler cluster thus expects all dependencies to be cached somewhere.
Make sure that BASIC_PATH in datasets_euler.py is set to point to the root of all datasets.

"""

import textattack
import sys
import torch
from models.BiRLM import BidirectionalGRUClassifier, BidirectionalLSTMClassifier
from torchtext.vocab import GloVe
from models.CNN import CNNClassifier, CNNClassifier2
from models.BERT import BERTClassifier
from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizer
from datasets_wrapped_ta import *
from attackutils.modelwrapper import CustomPyTorchModelWrapper, CustomBERTModelWrapper, CustomSEMModelWrapper

print('Prior to running this script on Euler: make sure to have the following environment variables set:')
print('export TA_CACHE_DIR=<PATH_TO_TEXTATTACK_CACHE>')
print('export TRANSFORMERS_OFFLINE=1')
print('export TRANSFORMERS_CACHE=<PATH_TO_TRANSFORMERS_LIB_CACHE>')
print('Also set BASIC_PATH in datasets_euler.py to the root of all datasets.')

if len(sys.argv) == 1:
    print(
        'Usage: python attack.py MODEL DATASET GLOVE_CACHE_PATH TRANSFORMERS_CACHE_PATH MODEL_PATH ATTACK_NAME CSV_PATH VERSION [SEM_EMBED_PATH]')
    print('Choices for MODEL: GRU, LSTM, CNN, BERT, CNN2')
    print('Choices for DATASET: IMDB, AG_NEWS, YahooAnswers')
    print('Choices for ATTACK_NAME: PWWS, BAE, FGA')
    print('Choices for VERSION: CLEAN, WLADL, SEM, PWWS')
    exit()
else:
    MODEL = sys.argv[1]
    DATASET = sys.argv[2]
    VECTOR_CACHE = sys.argv[3]
    TRANSFORMERS_CACHE = sys.argv[4]
    MODEL_PATH = sys.argv[5]
    ATTACK_NAME = sys.argv[6]
    CSV_PATH = sys.argv[7]
    VERSION = sys.argv[8]
    if VERSION == 'SEM':
        PATH_TO_SEM_EMBED = sys.argv[9]

print(f'Running with: {MODEL} {DATASET} {VECTOR_CACHE} {TRANSFORMERS_CACHE} {MODEL_PATH} {ATTACK_NAME} {CSV_PATH} {VERSION}')


model_name = MODEL + '_' + DATASET + '_' + VERSION

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if MODEL == 'BERT':
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True)
else:
    tokenizer = get_tokenizer('basic_english')

# get special textattack datasets
if DATASET == "AG_NEWS":
    dataset = get_textattack_AG_NEWS()
    num_classes = 4
elif DATASET == "IMDB":
    dataset = get_textattack_IMDB()
    num_classes = 2
elif DATASET == "YahooAnswers":
    dataset = get_textattack_YahooAnswers()
    num_classes = 10
else:
    raise ValueError()

if MODEL == 'GRU':
    model = BidirectionalGRUClassifier(num_classes, 64, 1).to(device)
elif MODEL == 'LSTM':
    model = BidirectionalLSTMClassifier(num_classes, 64, 1).to(device)
elif MODEL == 'CNN':
    model = CNNClassifier(num_classes, 1, [3, 5, 7], [2, 3, 4]).to(device)
elif MODEL == 'BERT':
    model = BERTClassifier(num_classes).to(device)
elif MODEL == 'CNN2':
    model = CNNClassifier2(num_classes).to(device)
else:
    raise ValueError()

if VERSION == 'SEM':
    embedding_path = PATH_TO_SEM_EMBED + \
        '/new_embeddings_d_3.1_k_10_' + DATASET + '.pt'
    embed = torch.load(embedding_path)
    checkpoint = torch.load(MODEL_PATH + '/' + model_name + '_3.1.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model_wrapper = CustomSEMModelWrapper(
        model, outdim=num_classes, vocab=embed, device=device)
else:
    checkpoint = torch.load(MODEL_PATH + '/' + model_name + '.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if MODEL == 'BERT':
        model_wrapper = CustomBERTModelWrapper(
            model, outdim=num_classes, device=device)
    else:
        model_wrapper = CustomPyTorchModelWrapper(
            model, outdim=num_classes, vocab=GloVe(name='6B', dim=50, cache=VECTOR_CACHE), device=device)

if ATTACK_NAME == 'PWWS':
    attack = textattack.attack_recipes.pwws_ren_2019.PWWSRen2019.build(
        model_wrapper)
elif ATTACK_NAME == 'BAE':
    attack = textattack.attack_recipes.bae_garg_2019.BAEGarg2019.build(
        model_wrapper)
elif ATTACK_NAME == 'FGA':
    attack = textattack.attack_recipes.faster_genetic_algorithm_jia_2019.FasterGeneticAlgorithmJia2019.build(
        model_wrapper)
else:
    raise ValueError()

# Attack 200 samples from the test sets
if ATTACK_NAME == 'FGA':
    # here we limit the number of queries because of the slow speed of GA
    attack_args = textattack.AttackArgs(
        num_examples=200, query_budget=5000, log_to_csv=CSV_PATH + '/' + model_name + '_' + ATTACK_NAME + '.csv')
else:
    attack_args = textattack.AttackArgs(
        num_examples=200, log_to_csv=CSV_PATH + '/' + model_name + '_' + ATTACK_NAME + '.csv')
attacker = textattack.Attacker(attack, dataset, attack_args)
attacker.attack_dataset()
