import torch
from torchtext.datasets import IMDB, AG_NEWS, YahooAnswers
from torchtext.vocab import GloVe
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
from dataset import ClassificationDataset
from models.BiRLM import BidirectionalGRUClassifier, BidirectionalLSTMClassifier

DATASET = 'AG_NEWS'
MODEL = 'LSTM'
VALIDATION_SPLIT = 0.5  # of test data
BATCH_SIZE = 64
SHUFFLE = True
NUM_EPOCHS = 10

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


model = BidirectionalGRUClassifier(num_classes, 64, 1).to(device)
optim = Adam(model.parameters())

for epoch in range(NUM_EPOCHS):
    train(model, optim, train_loader)
    val_accuracy = evaluate(model, val_loader)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'val_accuracy': val_accuracy
    }, PATH + '_' + str(epoch) + '.pt')

    # How to load a model
    #checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #epoch = checkpoint['epoch']
    #val_accuracy = checkpoint['val_accuracy']

test_accuracy = evaluate(model, test_loader)
print(f'Test accuracy: {test_accuracy}')
