# -*- coding: utf-8 -*-

import torch
from datasets_euler import AG_NEWS, IMDB, YahooAnswers
from torchtext.data import to_map_style_dataset

from torchtext.data.utils import get_tokenizer
from torch.nn import LSTM, GRU, Linear, Softmax, CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import math
from scipy.spatial import distance
import string
from torchtext.vocab import GloVe, vocab
import collections
from collections import Counter, OrderedDict
import pickle
import itertools
import sys
import timeit
import joblib

DATASET = 'AG_NEWS'  # choose from IMDB, AG_NEWS, YahooAnswers
MODEL = 'LSTM'  # choose from: GRU, LSTM, CNN, BERT, CNN2
VALIDATION_SPLIT = 0.5  # of test data
BATCH_SIZE = 64
SHUFFLE = True
NUM_EPOCHS = 5  # default 10

# This is the directory where glove.6B.50d.txt is stored
# You can download it at  https://www.kaggle.com/watts2/glove6b50dtxt?select=glove.6B.50d.txt
VECTOR_CACHE = '/cluster/scratch/$(USERNAME)/glove.6B'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embedding = GloVe(name='6B', dim=50, cache=VECTOR_CACHE)

# Class to handle the Dataset, it contains the tokenizer, the raw dataset and the
# number of classes of the Dataset. It is useful especially if handling with document
# classification tasks
class ClassificationDataset(Dataset):
    def __init__(self, dataset, num_classes, tokenizer, model):
        self.num_classes = num_classes
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        label, text = self.dataset.__getitem__(idx)
        if type(label) == str:
            if label == 'neg':
                label = 0
            else:
                label = 1
        else:
            label = int(label) - 1
        if self.model == 'BERT':
            return label, self.tokenizer(text, padding="max_length", return_tensors='pt', max_length=512, truncation=True)
        else:
            return label, self.tokenizer(text)

if DATASET == 'IMDB':
    train_set = IMDB(tokenizer, MODEL, split='train')
    test_set = IMDB(tokenizer, MODEL, split='test')
    num_classes = 2
elif DATASET == 'AG_NEWS':
    train_set = AG_NEWS(split='train')
    test_set = AG_NEWS(split='test')
    num_classes = 4
elif DATASET == 'YahooAnswers':
    train_set = YahooAnswers(tokenizer, MODEL, split='train')
    test_set = YahooAnswers(tokenizer, MODEL, split='test')
    num_classes = 10
else:
    raise ValueError()

if MODEL == 'BERT':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
else:
    tokenizer = get_tokenizer('basic_english')

train_set = to_map_style_dataset(train_set)
test_set = to_map_style_dataset(test_set)

train_set = ClassificationDataset(train_set, num_classes, tokenizer, MODEL)
test_set = ClassificationDataset(test_set, num_classes, tokenizer, MODEL)
test_set, val_set = random_split(test_set, [test_set.__len__() - int(VALIDATION_SPLIT * test_set.__len__(
)), int(VALIDATION_SPLIT * test_set.__len__())], generator=torch.Generator().manual_seed(42))

# Here we need the glove.6B.50d.txt file in this repository /content/drive/MyDrive/School/glove.6B/
# to work. The file can be downloaded at https://www.kaggle.com/watts2/glove6b50dtxt?select=glove.6B.50d.txt
# The function returns a counter object which contains a dictionary of all tokens in GloVe.
def make_glove_counter():
  i = 0
  glove_cnt = Counter()
  with open(VECTOR_CACHE + "glove.6B.50d.txt", 'r', encoding="utf-8") as f:
      for line in f:
        i+=1
        values = line.split()
        word = values[0]
        glove_cnt[word] = 0
  return glove_cnt

# Returns a counter object with the frequency of the tokens present in list_with_tokens
def make_frequency_counter(list_with_tokens):
  return Counter(list_with_tokens)

# Prints the first n keys in the dictionary
def print_first_n_key_val_dict(n, dictionary):
  cnt = 0
  for key, value in dictionary.items():
    cnt += 1
    print('Instance nr:', cnt)
    print('key is:')
    print(key)
    print()
    print('value is:')
    print(value)
    print()

    if(cnt == n):
      return

# Taking the train_set, the test_set and the val_set as we defined
# them in our code and return a counter object dictionary which contains
# the tokens and the number of times the token was used
def make_counter(train_set, test_set, val_set):
  counter_dict = Counter()

  for label, token_list in train_set:
    new_dict = make_frequency_counter(token_list)
    counter_dict.update(new_dict)

  for label, token_list in test_set:
    new_dict = make_frequency_counter(token_list)
    counter_dict.update(new_dict)

  for label, token_list in val_set:
    new_dict = make_frequency_counter(token_list)
    counter_dict.update(new_dict)

  return counter_dict

# Returns the eucledian distance between 2 word embeddings
def distance_between_tokens(token_1, token_2, embeddings):
  return distance.euclidean(embeddings[token_1], embeddings[token_2])

# Returns a sorted collection object [word, eucl_dist] with respect to increasing eucledian distance
# to the token called token_reference.
# We may assume that dict_with_tokens is a frequency ordered dictionary
def make_eucledian_distance_dict(token_reference, dict_with_tokens, embedding):
  eucledian_distance = OrderedDict()
  for token, frequency in dict_with_tokens.items():
    eucledian_distance[token] = distance_between_tokens(token, token_reference, embedding)
  return collections.OrderedDict(sorted(eucledian_distance.items(), key= lambda row: row[1]))

# Function returns a list with tuples. The maximal size of the list is nr_of_synonyms
# watch out that the dict_with_tokens has to be a dictionary element
def possible_synonyms(token_ref, dict_with_tokens, distance_for_sin, nr_of_synonyms, embeddings):
  dict_eucl_dist = make_eucledian_distance_dict(token_ref, dict_with_tokens, embedding)
  counter = 0
  synonyms = []
  # Loop is constructed in such a manner that it starts from the most similar word
  # which will always be itself
  for token, eucledian_distance in dict_eucl_dist.items():
    counter += 1
    if(eucledian_distance <= distance_for_sin):
      if (token != token_ref):
        synonyms.append((token, eucledian_distance))
    else:
      break

    if(counter >= (nr_of_synonyms+1)):
      break

  dict_eucl_dist.clear()
  return synonyms


# Returns a dictionary with the new encodings according to the SEM Algorithm
# I may assume the dict_of_words to be the frequency counter object sorted by frequency
def Synonym_Encoding_Algorithm(top_nr_of_tks_dict, distance, nr_of_synonyms, \
                               embedding, glove_counter, return_tot_emb):

  #Setting the value of the keys to NULL in order to flag them
  for token, frequency in top_nr_of_tks_dict.items():
    top_nr_of_tks_dict[token] = 'NULL'

  counter = 0
  tot_time = 0

  print("Starting SEM")
  for token in top_nr_of_tks_dict:
    counter += 1

    # Self made function to look at progress while running
    if(counter % 1000 == 0):
      print(counter,' out of ', len(top_nr_of_tks_dict))
      print('Expected total time is: ', len(top_nr_of_tks_dict)*tot_time/counter)

    starttime = timeit.default_timer()
    synonym_list = possible_synonyms(token, \
              top_nr_of_tks_dict, distance, nr_of_synonyms, embedding)
    endtime = timeit.default_timer()
    tot_time += (endtime - starttime)

    if (top_nr_of_tks_dict[token] == 'NULL'):
      # Looping through the synonyms
      loop_counter = 0
      for synonym_token, similarity in synonym_list:
        loop_counter += 1

        if (top_nr_of_tks_dict[synonym_token] != 'NULL'):
          loop_counter -= 1
          top_nr_of_tks_dict[token] = top_nr_of_tks_dict[synonym_token]
          break

      # Case if we didn't found encoded synonyms
      if(loop_counter == len(synonym_list)):
        top_nr_of_tks_dict[token] = embedding[token]
    synonym_list.clear()

  # Here we incorporate the new embeddings in the old input embedding
  # or we just return the embeddings of the top nr_of_tks words
  if(return_tot_emb):
    new_embedding = dict()
    for token, _ in glove_counter.items():
      new_embedding[token] = embedding[token]
    new_embedding.update(top_nr_of_tks_dict)
    return new_embedding
  else:
    return top_nr_of_tks_dict

print('Preparing our counter objects')
glove_counter = make_glove_counter()
dataset_counter = make_counter(train_set, test_set, val_set)

# We integrate the counter object from the dataset into our glove counter
glove_counter.update(dataset_counter)

# This variable defines the nr of words we input into SEM. It scales quadratically so beware
nr_of_tks = 50000
top_nr_of_tks_list = glove_counter.most_common()[0:nr_of_tks]
top_nr_of_tks_dict = dict(top_nr_of_tks_list)

# max_euclidian_distance determines how big the euclidian distance maximally is
# between the embedding of 2 tokens in order to be considered similar.
# The max_nr_of_synonyms determines the maximal number of synonyms a word can have
max_euclidian_distance = 3.6
max_nr_of_synonyms = 10

print('Starting with a SEM')
new_embedding = Synonym_Encoding_Algorithm(top_nr_of_tks_dict, \
  max_euclidian_distance, max_nr_of_synonyms, embedding, glove_counter, False)

# Here we define the PATH were the new embeddings will be stored
# You may adapt PATH to your liking
PATH='./'
dictionary_name = 'new_embeddings_d_%.1f_k_%d.pt'%(max_euclidian_distance, max_nr_of_synonyms)

torch.save(new_embedding, PATH + 'small_' +dictionary_name)

print("New_Embeddings list saved in " + PATH)