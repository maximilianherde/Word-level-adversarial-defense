import textattack
import sys
import torch
from models.BiRLM import BidirectionalGRUClassifier, BidirectionalLSTMClassifier
from models.CNN import CNNClassifier, CNNClassifier2
from models.BERT import BERTClassifier
from attackutils.modelwrapper import CustomPyTorchModelWrapper, CustomBERTModelWrapper

MODEL = 'CNN'
MODEL_PATH = '~'
DATASET = 'AG_NEWS'
ATTACK_NAME = 'PWWS'
CSV_PATH = '~'
VERSION = 'CLEAN'

model_name = MODEL + '_' + DATASET + '_' + VERSION

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DATASET == "AG_NEWS":
    dataset = textattack.datasets.HuggingFaceDataset("ag_news", split="test")
    num_classes = 4
elif DATASET == "IMDB":
    dataset = textattack.datasets.HuggingFaceDataset("imdb", split="test")
    num_classes = 2
elif DATASET == "YahooAnswers":
    dataset = textattack.datasets.HuggingFaceDataset(
        "yahoo_answers", split="test")
    num_classes = 10
else:
    raise ValueError()

if MODEL == 'GRU':
    model = BidirectionalGRUClassifier(num_classes, 64, 1).to(device)
elif MODEL == 'LSTM':
    model = BidirectionalLSTMClassifier(num_classes, 64, 1).to(device)
elif MODEL == 'CNN':
    model = CNNClassifier(num_classes, 1, [3, 5, 7], [2, 3, 4]).to(device)
    # todo: ADD right parameters: num_classes, in_channels, out_channels, kernel_heights
elif MODEL == 'BERT':
    model = BERTClassifier(num_classes).to(device)
elif MODEL == 'CNN2':
    model = CNNClassifier2(num_classes).to(device)
else:
    raise ValueError()

checkpoint = torch.load(MODEL_PATH + '/' + model_name + '.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

if MODEL == 'BERT':
    model_wrapper = CustomBERTModelWrapper(model, outdim=num_classes)
else:
    model_wrapper = CustomPyTorchModelWrapper(model, outdim=num_classes)

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

attack_args = textattack.AttackArgs(
    num_examples=200, log_to_csv=CSV_PATH + '/' + model_name + '_' + ATTACK_NAME + '.csv')
attacker = textattack.Attacker(attack, dataset, attack_args)
attacker.attack_dataset()
