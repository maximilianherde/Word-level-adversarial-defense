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
from datasets_euler import AG_NEWS, IMDB, YahooAnswers
import time

DATASET = 'YahooAnswers'  # choose from IMDB, AG_NEWS, YahooAnswers
MODEL = 'CNN'  # choose from: GRU, LSTM, CNN, BERT, CNN2
VALIDATION_SPLIT = 0.5  # of test data
BATCH_SIZE = 256
SHUFFLE = True
NUM_EPOCHS = 5  # default 10
PATH = './checkpoints/'
TRAIN = True
CHECKPOINT = 0  # last CHECKPOINT = NUM_EPOCHS - 1
MAX_LEN_BERT = 300
VECTOR_CACHE = '/cluster/scratch/herdem/glove'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if MODEL == 'BERT':
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True)
else:
    tokenizer = get_tokenizer('basic_english')

if DATASET == 'IMDB':
    train_set = IMDB(tokenizer, MODEL, split='train')
    test_set = IMDB(tokenizer, MODEL, split='test')
    num_classes = 2
elif DATASET == 'AG_NEWS':
    train_set = AG_NEWS(tokenizer, MODEL, split='train')
    test_set = AG_NEWS(tokenizer, MODEL, split='test')
    num_classes = 4
elif DATASET == 'YahooAnswers':
    train_set = YahooAnswers(tokenizer, MODEL, split='train')
    test_set = YahooAnswers(tokenizer, MODEL, split='test')
    num_classes = 10
else:
    raise ValueError()

embedding = GloVe(name='6B', dim=50, cache=VECTOR_CACHE)

#train_set = to_map_style_dataset(train_set)
#test_set = to_map_style_dataset(test_set)

#train_set = ClassificationDataset(train_set, num_classes, tokenizer, MODEL)
#test_set = ClassificationDataset(test_set, num_classes, tokenizer, MODEL)
test_set, val_set = random_split(test_set, [test_set.__len__() - int(VALIDATION_SPLIT * test_set.__len__(
)), int(VALIDATION_SPLIT * test_set.__len__())], generator=torch.Generator().manual_seed(42))


def collate_batch(batch):
    label_list, text_list = [], []
    for (_label, _tokens) in batch:
        label_list.append(_label)
        embed = embedding.get_vecs_by_tokens(_tokens)
        text_list.append(embed)
    text_list = pad_sequence(text_list, batch_first=True)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return label_list.to(device), text_list.to(device)


def collate_BERT(batch):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_acc += (output.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
            pbar.update()
            if idx % log_interval == 0 and idx > 0:
                pbar.set_postfix(loss=loss_, accuracy=total_acc / total_count)
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
            pbar.update()
            if idx % log_interval == 0 and idx > 0:
                pbar.set_postfix(loss=loss_, accuracy=total_acc / total_count)
                total_acc, total_count = 0, 0

        pbar.close()


if MODEL == 'GRU':
    model = BidirectionalGRUClassifier(num_classes, 64, 1).to(device)
    optim = Adam(model.parameters())
elif MODEL == 'LSTM':
    model = BidirectionalLSTMClassifier(num_classes, 64, 1).to(device)
    optim = Adam(model.parameters())
elif MODEL == 'CNN':
    model = CNNClassifier(num_classes, 1, [3, 5, 7], [2, 3, 4]).to(device)
    # todo: ADD right parameters: num_classes, in_channels, out_channels, kernel_heights
    optim = Adam(model.parameters())
elif MODEL == 'BERT':
    model = BERTClassifier(num_classes).to(device)
    optim = AdamW(model.parameters(), lr=3e-5, correct_bias=False)
elif MODEL == 'CNN2':
    model = CNNClassifier2(num_classes).to(device)
    optim = Adam(model.parameters())

if TRAIN:
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        train(model, optim, train_loader)
        val_accuracy = evaluate(model, val_loader)
        # creates checkpoints directory if not existing yet
        Path("./checkpoints").mkdir(parents=True, exist_ok=True)
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

start_time_test = time.time()

stats_ = stats(model, MODEL, test_loader, avg="weighted")
print(f'Test stats (accuracy, RocAuc, f1): {stats_}')

end_time_test = time.time()
print("elapsed testing time (min): ", round(
    (end_time_test - start_time_test)/60.0, 3))

# test_accuracy = evaluate(model, test_loader)
# print(f'Test accuracy: {test_accuracy}')

# # testing metrics
# print(f'Test accuracy: {accuracy(model,MODEL,test_loader)}')
# print(f'Test auroc: {auroc(model,MODEL,test_loader, avg="weighted")}')
# print(f'Test f1: {f1(model,MODEL,test_loader, avg="weighted")}')

# # all in one statistics: accuracy, roc-auc and f1
# stats_ = stats(model, MODEL, test_loader, avg="weighted")

# print(f'Test stats (accuracy, RocAuc, f1): {stats_}')
