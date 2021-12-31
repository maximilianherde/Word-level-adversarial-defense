"""

Everything you need to defend using WLADL.
WLADL layer in mask_replace_with_syns_add_noise.

"""

import torch
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from tqdm import tqdm


def build_thesaurus(all_words, pbar_update=True):
    """
    Build the synonym thesaurus.
    """
    # Expects torch.vocab.itos dictionary to extract thesaurus
    thesaurus = {}
    syns = []
    length_thesaurus = len(all_words)

    pbar = tqdm(total=length_thesaurus, desc='Build thesaurus')
    for i in range(length_thesaurus):
        # Extract the word
        token = all_words[i]

        # Find the synsets for the token
        synsets = wn.synsets(token)

        if len(synsets) == 0:
            thesaurus[token] = ""

        else:
            # Iterate through all synset
            for synset in synsets:
                lemma_names = synset.lemma_names()
                for lemma in lemma_names:
                    # Check if lemma has an underscore indicating a two word token
                    if not("_" in lemma):
                        lemma = lemma.lower()
                        if (lemma != token and lemma not in syns):
                            syns.append(lemma)

            thesaurus[token] = syns
            syns = []
        if pbar_update:
            pbar.update()

    pbar.close()
    return thesaurus


def mask_replace_with_syns_add_noise(sentence, thesaurus, embedding, model, mask_probability=0.1, synonym_probability=0.25, pos_noise=0.1):
    """
    Word-level Adversarial Defense Layer
    Expects sentence, a synonym thesaurus, embedding and a model specifier. Set $p_1$ and $p_2$ using mask_probability and synonym_probability.
    Returns embedding vectors for all models except BERT for which it returns tokens.
    """
    tokens_to_ret = []
    for word in sentence:
        mask_flag = np.random.choice([0, 1], replace=False, p=[
                                     1-mask_probability, mask_probability])
        # Not masked
        if mask_flag == 0:
            syn_flag = np.random.choice([0, 1], replace=False, p=[
                                        1-synonym_probability, synonym_probability])
            # Not masked & replaced with synonym
            if syn_flag == 1:
                # Check if word exists in thesaurus
                synonyms = thesaurus.get(word)
                if synonyms != None and len(synonyms) != 0:
                    # randomly sample a synonym word
                    indx = np.random.randint(low=0, high=len(synonyms))
                    tokens_to_ret.append(synonyms[indx])
                # Synonym doesn't exist
                else:
                    tokens_to_ret.append(word)
            # Not masked & not replaced with synonym
            else:
                tokens_to_ret.append(word)
        # Masked
        else:
            tokens_to_ret.append("")

    if model == 'BERT':
        embed = tokens_to_ret
    else:
        # We have masked and replaced with synonyms randomly, now obtain embeddings
        embed = embedding.get_vecs_by_tokens(tokens_to_ret)

    '''pos_encoding = np.zeros(embed.shape)
    # Positional encoding introduced in Vaswani et. al.
    for i in range(embed.shape[0]):
        if i%2 == 0:
            pos_param = pos_noise*np.sin(i / (10000 ** ((2*(i//2) / embed.shape[1]))))
        else:
            pos_param = pos_noise*np.cos(i / (10000 ** ((2*(i//2) / embed.shape[1]))))'''

    return embed
