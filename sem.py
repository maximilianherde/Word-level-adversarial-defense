# -*- coding: utf-8 -*-
"""SEM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14VcbO1Tp5KIzXAMQdcZna4NnbqhjVFJv
"""

!pip install transformers

import torch
from torchtext.datasets import IMDB, AG_NEWS, YahooAnswers
#from datasets_euler import AG_NEWS, IMDB, YahooAnswers
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
VECTOR_CACHE = '/cluster/scratch/noec/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from google.colab import drive
drive.mount('/content/drive')
PATH = '/content/drive/MyDrive/Checkpoints/School'
filename = 'embedding.pt'
embedding = torch.load(PATH + filename)

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

tokenizer = get_tokenizer('basic_english')

if DATASET == 'IMDB':
    train_set = IMDB(tokenizer, MODEL, split='train')
    test_set = IMDB(tokenizer, MODEL, split='test')
    num_classes = 2
elif DATASET == 'AG_NEWS':
    train_set = AG_NEWS(split='train')#(tokenizer, MODEL, split='train')
    test_set = AG_NEWS(split='test')#(tokenizer, MODEL, split='test')
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

def save_dictionary_to(PATH, dictionary_name, dictionary_to_save):
  a_file = open(PATH + dictionary_name, 'wb')
  joblib.dump(dictionary_to_save, a_file)
  a_file.close()

def return_dictionary_from(PATH, dictionary_name):
  a_file = PATH + dictionary_name
  output = joblib.load(a_file)
  return output

def save_torch_to(PATH, torch_name, torch_to_save):
  savedest = PATH + torch_name
  torch.save(torch_to_save, savedest)

def return_torch_from(PATH, dictionary_name):
  a_file = PATH + dictionary_name
  output = torch.load(a_file)
  return output

def make_glove_counter():
  PATH = '/content/drive/MyDrive/School/glove.6B/'
  i = 0
  glove_cnt = Counter()
  with open(PATH + "glove.6B.50d.txt", 'r', encoding="utf-8") as f:
      for line in f:
        i+=1
        values = line.split()
        word = values[0]
        glove_cnt[word] = 0
  return glove_cnt

#returns a counter object with the frequency of the tokens present in list_with_tokens
def make_frequency_counter(list_with_tokens):
  return Counter(list_with_tokens)

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

#returns the eucledian distance between 2 word embeddings
def distance_between_tokens(token_1, token_2, embeddings):
  return distance.euclidean(embeddings[token_1], embeddings[token_2])

#returns a sorted collection object [word, eucl_dist] with respect to increasing eucl_dist
#dict_with_tokens is a dictionary frequency_ordered
def make_eucledian_distance_dict(token_reference, dict_with_tokens, embedding):
  eucledian_distance = OrderedDict()
  for token, frequency in dict_with_tokens.items():
    eucledian_distance[token] = distance_between_tokens(token, token_reference, embedding)
  return collections.OrderedDict(sorted(eucledian_distance.items(), key= lambda row: row[1]))

#this returns a list with tuples. maximal size of the list is nr_of_synonyms
# watch out that the dict_with_tokens has to be a dictionary element
def possible_synonyms(token_ref, dict_with_tokens, distance_for_sin, nr_of_synonyms, embeddings):
  dict_eucl_dist = make_eucledian_distance_dict(token_ref, dict_with_tokens, embedding)
  counter = 0
  synonyms = []
  #code is constructed in such a manner that it starts from the most similar word
  #which will always be itself
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


#Out is a dictionary with the encoding results
# I may assume the dict_of_words to be the frequency counter object sorted by frequency
#encoding is a frequency ordered count dictionary 
def Synonym_Encoding_Algorithm(top_50000_dict, distance, nr_of_synonyms, \
                               embedding, glove_dict, return_tot_emb):

  #Setting the value of the keys to NULL in order to flag them
  for token, frequency in top_50000_dict.items():
    top_50000_dict[token] = 'NULL'

  counter = 0
  tot_time = 0

  print("Starting SEM")
  for token in top_50000_dict:
    counter += 1
    if(counter % 10 == 0):
      print(counter,' out of ', len(top_50000_dict))
      print('Expected total time is: ', 50000*tot_time/counter)

    starttime = timeit.default_timer()
    synonym_list = possible_synonyms(token, \
              top_50000_dict, distance, nr_of_synonyms, embedding)
    endtime = timeit.default_timer()
    tot_time += (endtime - starttime)

    if (top_50000_dict[token] == 'NULL'):
      #Looping through the synonyms
      loop_counter = 0
      for synonym_token, similarity in synonym_list:
        loop_counter += 1

        if (top_50000_dict[synonym_token] != 'NULL'):
          loop_counter -= 1
          top_50000_dict[token] = top_50000_dict[synonym_token]
          break

      #case if we didn't found encoded synonyms
      if(loop_counter == len(synonym_list)):
        top_50000_dict[token] = embedding[token]
    synonym_list.clear()

  #Creation of new_embedding dictionary with the hepl of embedding
  if(return_tot_emb):
    new_embedding = dict()
    for token, _ in glove_dict.items():
      new_embedding[token] = embedding[token]
    new_embedding.update(top_50000_dict)
    return new_embedding
  else:
    return top_50000_dict

print('starting with functions')
glove_dict = make_glove_counter()
ds_cnt = make_counter(train_set, test_set, val_set)
glove_dict.update(ds_cnt)
top_n = 50000 #50000
top_50000_list = glove_dict.most_common()[0:top_n]
top_50000_dict = dict(top_50000_list)
max_euclidian_distance = 3.6
max_nr_of_synonyms = 10

print('starting with a new embedding')
new_embedding = Synonym_Encoding_Algorithm(top_50000_dict, \
  max_euclidian_distance, max_nr_of_synonyms, embedding, glove_dict, False)

PATH = '/content/drive/MyDrive/School/'
dictionary_name = 'new_embeddings_d_%.1f_k_%d.pt'%(max_euclidian_distance, max_nr_of_synonyms)

torch.save(new_embedding, PATH + 'small_' +dictionary_name)

print("New_Embeddings list saved in " + PATH)