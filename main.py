"""

Main driver for training all models with or without WLADL. Can also resume training from checkpoints.
It is thought to be used in a firewalled environment such as the compute nodes of ETHZ's Euler cluster thus expects all dependencies to be cached somewhere.
Saves a checkpoint for every epoch in ./checkpoints

"""

import torch
# from torchtext.datasets import IMDB, AG_NEWS, YahooAnswers
from torchtext.vocab import GloVe
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
from dataset import ClassificationDataset
from models.BiRLM import BidirectionalGRUClassifier, BidirectionalLSTMClassifier
from models.CNN import CNNClassifier, CNNClassifier2
from models.BERT import BERTClassifier
from metrics.accuracy import accuracy
from metrics.auroc import auroc
from metrics.f1 import f1
from metrics.stats import stats
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from pathlib import Path
from datasets_euler import AG_NEWS, IMDB, YahooAnswers, YahooAnswers_ADV
import time
import os
import sys
from WLADL import build_thesaurus, mask_replace_with_syns_add_noise

print('Prior to running this script on Euler: make sure to have the following environment variables set:')
print('export TRANSFORMERS_OFFLINE=1')
print('export TRANSFORMERS_CACHE=<PATH_TO_TRANSFORMERS_LIB_CACHE>')

if len(sys.argv) == 1:
    print(
        'Usage: python main.py MODEL DATASET BATCH_SIZE GLOVE_CACHE_PATH TRANSFORMERS_CACHE_PATH ON_CLUSTER CHECKPOINT TRAIN [NUM_EPOCHS] [WITH_DEFENSE]')
    print('Choices for MODEL: GRU, LSTM, CNN, BERT, CNN2')
    print('Choices for DATASET: IMDB, AG_NEWS, YahooAnswers, YahooAnswers_ADV')
    exit()
else:
    MODEL = sys.argv[1]
    DATASET = sys.argv[2]
    BATCH_SIZE = int(sys.argv[3])
    VECTOR_CACHE = sys.argv[4]
    TRANSFORMERS_CACHE = sys.argv[5]
    if sys.argv[6] == 'True':
        CLUSTER = True
    else:
        CLUSTER = False
    # 0 if training from scratch, last CHECKPOINT = NUM_EPOCHS - 1
    CHECKPOINT = int(sys.argv[7])
    if sys.argv[8] == 'True':
        TRAIN = True
        NUM_EPOCHS = int(sys.argv[9])
        if sys.argv[10] == 'True':
            WITH_DEFENSE = True
        else:
            WITH_DEFENSE = False
    else:
        TRAIN = False
        NUM_EPOCHS = 0
        WITH_DEFENSE = False

PATH = './checkpoints/'
VALIDATION_SPLIT = 0.5  # of test data
SHUFFLE = True
MAX_LEN_BERT = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if MODEL == 'BERT':
    # if CLUSTER:
    #    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    #    os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True)
else:
    tokenizer = get_tokenizer('basic_english')

# load the GloVe embeddings from a specified cache
embedding = GloVe(name='6B', dim=50, cache=VECTOR_CACHE)
example_thes = None
if WITH_DEFENSE:
    # build WordNet thesaurus for WLADL
    example_thes = build_thesaurus(embedding.itos, not CLUSTER)

# get the datsets
if DATASET == 'IMDB':
    train_set = IMDB(tokenizer, MODEL, split='train', with_defense=WITH_DEFENSE,
                     thesaurus=example_thes, embedding=embedding)
    test_set = IMDB(tokenizer, MODEL, split='test', with_defense=WITH_DEFENSE,
                    thesaurus=example_thes, embedding=embedding)
    num_classes = 2
elif DATASET == 'AG_NEWS':
    train_set = AG_NEWS(tokenizer, MODEL, split='train',
                        with_defense=WITH_DEFENSE, thesaurus=example_thes, embedding=embedding)
    test_set = AG_NEWS(tokenizer, MODEL, split='test', with_defense=WITH_DEFENSE,
                       thesaurus=example_thes, embedding=embedding)
    num_classes = 4
elif DATASET == 'YahooAnswers':
    train_set = YahooAnswers(tokenizer, MODEL, split='train',
                             with_defense=WITH_DEFENSE, thesaurus=example_thes, embedding=embedding)
    test_set = YahooAnswers(tokenizer, MODEL, split='test',
                            with_defense=WITH_DEFENSE, thesaurus=example_thes, embedding=embedding)
    num_classes = 10
elif DATASET == 'YahooAnswers_ADV':
    train_set = YahooAnswers_ADV(tokenizer, MODEL, split='train',
                                 with_defense=WITH_DEFENSE, thesaurus=example_thes, embedding=embedding)
    test_set = YahooAnswers_ADV(tokenizer, MODEL, split='test',
                                with_defense=WITH_DEFENSE, thesaurus=example_thes, embedding=embedding)
    num_classes = 10
else:
    raise ValueError()

# train_set = to_map_style_dataset(train_set)
# test_set = to_map_style_dataset(test_set)

# train_set = ClassificationDataset(train_set, num_classes, tokenizer, MODEL)
# test_set = ClassificationDataset(test_set, num_classes, tokenizer, MODEL)
test_set, val_set = random_split(test_set, [test_set.__len__() - int(VALIDATION_SPLIT * test_set.__len__(
)), int(VALIDATION_SPLIT * test_set.__len__())], generator=torch.Generator().manual_seed(42))


def collate_batch(batch):
    """
    expects tokens, gets embedding vectors, then pads and returns labels and text
    for use with BiLSTM/BiGRU and CNN
    """
    label_list, text_list = [], []
    for (_label, _tokens) in batch:
        label_list.append(_label)
        embed = embedding.get_vecs_by_tokens(_tokens)
        text_list.append(embed)
    text_list = pad_sequence(text_list, batch_first=True)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return label_list.to(device), text_list.to(device)


def collate_defense_batch(batch):
    """
    collate fn for WLADL application
    applies the defense layer which returns embeddings, then pads and returns labels and text
    """
    label_list, text_list = [], []
    for (_label, _tokens) in batch:
        label_list.append(_label)
        embed = mask_replace_with_syns_add_noise(
            _tokens, example_thes, embedding, MODEL)
        text_list.append(embed)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True)
    return label_list.to(device), text_list.to(device)


def collate_BERT(batch):
    """
    returns everything that BERT needs
    """
    label_list, input_ids, token_type_ids, attention_mask = [], [], [], []
    for (_label, _dic) in batch:
        label_list.append(_label)
        input_ids.append(_dic['input_ids'])
        token_type_ids.append(_dic['token_type_ids'])
        attention_mask.append(_dic['attention_mask'])
    label_list = torch.tensor(label_list, dtype=torch.int64).to(device)
    input_ids = torch.cat(input_ids, dim=0).to(device)
    token_type_ids = torch.cat(token_type_ids, dim=0).to(device)
    attention_mask = torch.cat(attention_mask, dim=0).to(device)
    return label_list, input_ids, token_type_ids, attention_mask


if WITH_DEFENSE:
    if MODEL == 'BERT':
        train_loader = DataLoader(
            train_set, batch_size=BATCH_SIZE, collate_fn=collate_BERT, shuffle=SHUFFLE)
        test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, collate_fn=collate_BERT, shuffle=SHUFFLE)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                                collate_fn=collate_BERT, shuffle=SHUFFLE)
    else:
        train_loader = DataLoader(
            train_set, batch_size=BATCH_SIZE, collate_fn=collate_defense_batch, shuffle=SHUFFLE)
        test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, collate_fn=collate_defense_batch, shuffle=SHUFFLE)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                                collate_fn=collate_defense_batch, shuffle=SHUFFLE)
else:
    if MODEL == 'BERT':
        train_loader = DataLoader(
            train_set, batch_size=BATCH_SIZE, collate_fn=collate_BERT, shuffle=SHUFFLE)
        test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, collate_fn=collate_BERT, shuffle=SHUFFLE)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                                collate_fn=collate_BERT, shuffle=SHUFFLE)
    else:
        train_loader = DataLoader(
            train_set, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=SHUFFLE)
        test_loader = DataLoader(
            test_set, batch_size=BATCH_SIZE, collate_fn=collate_batch, shuffle=SHUFFLE)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                                collate_fn=collate_batch, shuffle=SHUFFLE)


def evaluate(model, data_loader, loss=CrossEntropyLoss()):
    """
    Evaluates the model on given data and returns accuracy on it.
    """
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        if MODEL == "BERT":
            for idx, (labels, input_ids, token_type_ids, attention_mask) in enumerate(data_loader):
                predicted_label = model(
                    input_ids, token_type_ids, attention_mask)
                loss_ = loss(predicted_label, labels)
                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)
        else:
            for idx, (labels, text) in enumerate(data_loader):
                predicted_label = model(text)
                loss_ = loss(predicted_label, labels)
                total_acc += (predicted_label.argmax(1) == labels).sum().item()
                total_count += labels.size(0)

    return total_acc / total_count


def train(model, optimizer, train_loader, loss=CrossEntropyLoss(), log_interval=50):
    """
    Trains the model using optimizer on the given dataset.
    Progress bar can be controlled using global CLUSTER param.
    """
    model.train()
    total_acc, total_count = 0, 0
    pbar = tqdm(total=len(train_loader),
                desc=f'Epoch [{epoch + 1}/{NUM_EPOCHS}]')

    if MODEL == 'BERT':
        for idx, (labels, input_ids, token_type_ids, attention_mask) in enumerate(train_loader):
            output = model(input_ids, token_type_ids, attention_mask)
            loss_ = loss(output, labels)
            optimizer.zero_grad()
            loss_.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)  # don't let grads explode
            optimizer.step()
            total_acc += (output.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
            if not CLUSTER:
                pbar.update()
            if idx % log_interval == 0 and idx > 0:
                if not CLUSTER:
                    pbar.set_postfix(
                        loss=loss_, accuracy=total_acc / total_count)
                total_acc, total_count = 0, 0

        pbar.close()
    else:
        for idx, (labels, text) in enumerate(train_loader):
            output = model(text)
            loss_ = loss(output, labels)
            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()
            total_acc += (output.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
            if not CLUSTER:
                pbar.update()
            if idx % log_interval == 0 and idx > 0:
                if not CLUSTER:
                    pbar.set_postfix(
                        loss=loss_, accuracy=total_acc / total_count)
                total_acc, total_count = 0, 0

        pbar.close()


# select model, always train using Adam and default params.
if MODEL == 'GRU':
    model = BidirectionalGRUClassifier(num_classes, 64, 1).to(device)
    optim = Adam(model.parameters())
elif MODEL == 'LSTM':
    model = BidirectionalLSTMClassifier(num_classes, 64, 1).to(device)
    optim = Adam(model.parameters())
elif MODEL == 'CNN':
    model = CNNClassifier(num_classes, 1, [3, 5, 7], [2, 3, 4]).to(device)
    optim = Adam(model.parameters())
elif MODEL == 'BERT':
    model = BERTClassifier(num_classes).to(device)
    optim = AdamW(model.parameters(), lr=3e-5, correct_bias=False)
elif MODEL == 'CNN2':
    model = CNNClassifier2(num_classes).to(device)
    optim = Adam(model.parameters())
else:
    raise ValueError()

if TRAIN:
    epochs_already_done = 0
    if CHECKPOINT != 0:
        # load from checkpoint then train
        # load a pretrained model
        PATH_MODEL = PATH + MODEL + '_' + \
            DATASET + '_' + str(CHECKPOINT) + '.pt'
        checkpoint = torch.load(PATH_MODEL)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs_already_done = checkpoint['epoch'] + 1
        val_accuracy = checkpoint['val_accuracy']
    # creates checkpoints directory if not existing yet
    Path(PATH).mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    for epoch in range(epochs_already_done, NUM_EPOCHS + epochs_already_done):
        train(model, optim, train_loader)
        val_accuracy = evaluate(model, val_loader)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'val_accuracy': val_accuracy
        }, PATH + MODEL + '_' + DATASET + '_' + str(epoch) + '.pt')
    end_time = time.time()
    print("elapsed training time (min): ",
          round((end_time - start_time)/60.0, 3))
else:
    # load a pretrained model
    PATH_MODEL = PATH + MODEL + '_' + DATASET + '_' + str(CHECKPOINT) + '.pt'
    checkpoint = torch.load(PATH_MODEL)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_accuracy = checkpoint['val_accuracy']

print("Model :" + str(MODEL) + ", Dataset: " + str(DATASET) +
      ", Epochs: " + str(NUM_EPOCHS if TRAIN else (CHECKPOINT + 1)))

# all in one statistics: accuracy, roc-auc and f1
stats_t = stats(model, MODEL, train_loader, avg="weighted")
print(f'Train stats (accuracy, RocAuc, f1): {stats_t}')

# test model and output stats
start_time_test = time.time()

stats_ = stats(model, MODEL, test_loader, avg="weighted")
print(f'Test stats (accuracy, RocAuc, f1): {stats_}')

end_time_test = time.time()
print("elapsed testing time (min): ", round(
    (end_time_test - start_time_test)/60.0, 3))
