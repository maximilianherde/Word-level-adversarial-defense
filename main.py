import torch
from torchtext.datasets import IMDB, AG_NEWS, YahooAnswers
from torchtext.vocab import GloVe
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from pytorch_pretrained_bert import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
from dataset import ClassificationDataset
from models.BiRLM import BidirectionalGRUClassifier, BidirectionalLSTMClassifier
from models.CNN import CNNClassifier, CNNClassifier2
from models.BERT import BERTClassifier
from pathlib import Path
import time

DATASET = 'AG_NEWS'  # choose from IMDB, AG_NEWS, YahooAnswers
MODEL = 'CNN2'  # choose from: GRU, LSTM, CNN, BERT, CNN2
VALIDATION_SPLIT = 0.5  # of test data
BATCH_SIZE = 64
SHUFFLE = True
NUM_EPOCHS = 2  # default 10
PATH = './checkpoints/'
TRAIN = False
CHECKPOINT = 1  # last CHECKPOINT = NUM_EPOCHS - 1
MAX_LEN_BERT = 300

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DATASET == 'IMDB':
    train_set = IMDB(split='train')
    test_set = IMDB(split='test')
    num_classes = 2
elif DATASET == 'AG_NEWS':
    train_set = AG_NEWS(split='train')
    test_set = AG_NEWS(split='test')
    num_classes = 4
elif DATASET == 'YahooAnswers':
    train_set = YahooAnswers(split='train')
    test_set = YahooAnswers(split='test')
    num_classes = 10
else:
    raise ValueError()

if MODEL == 'BERT':
    tokenizer = None
    tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
else:
    tokenizer = get_tokenizer('basic_english')

embedding = GloVe(name='6B', dim=50)

train_set = to_map_style_dataset(train_set)
test_set = to_map_style_dataset(test_set)

train_set = ClassificationDataset(train_set, num_classes, tokenizer)
test_set = ClassificationDataset(test_set, num_classes, tokenizer)
test_set, val_set = random_split(test_set, [test_set.__len__() - int(VALIDATION_SPLIT * test_set.__len__(
)), int(VALIDATION_SPLIT * test_set.__len__())], generator=torch.Generator().manual_seed(42))


def collate_batch(batch):
    label_list, text_list = [], []
    if MODEL == 'BERT':
        for (_label, _tokens) in batch:
            label_list.append(_label)
            # toDo: proper pre-processing for BERT model
            sentences = ["[CLS] " + s for s in _tokens]
            tokenized_sentences = [tokenizer_bert.tokenize(s) for s in sentences]
            tokenized_cropped = [t[:(MAX_LEN_BERT - 1)] + ['SEP'] for t in tokenized_sentences]
            embed = 0
            # toDo: proper pre-processing for BERT model
            text_list.append(embed)
    else:
        for (_label, _tokens) in batch:
            label_list.append(_label)
            embed = embedding.get_vecs_by_tokens(_tokens)
            text_list.append(embed)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True)
    return label_list.to(device), text_list.to(device)


train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          collate_fn=collate_batch, shuffle=SHUFFLE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                         collate_fn=collate_batch, shuffle=SHUFFLE)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                        collate_fn=collate_batch, shuffle=SHUFFLE)


def evaluate(model, data_loader, loss=CrossEntropyLoss()):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
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
    model = CNNClassifier(num_classes, 1, [2, 2, 2], [3, 3, 3]).to(device)
    # todo: ADD right parameters: num_classes, in_channels, out_channels, kernel_heights
    optim = Adam(model.parameters())
elif MODEL == 'BERT':
    model = BERTClassifier(num_classes).to(device)
    optim = Adam(model.parameters())
elif MODEL == 'CNN2':
    model = CNNClassifier2(num_classes).to(device)
    optim = Adam(model.parameters())

if TRAIN:
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
else:
    # load a pretrained model
    PATH_MODEL = PATH + MODEL + '_' + DATASET + '_' + str(CHECKPOINT) + '.pt'
    checkpoint = torch.load(PATH_MODEL)
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_accuracy = checkpoint['val_accuracy']

test_accuracy = evaluate(model, test_loader)
print(f'Test accuracy: {test_accuracy}')
