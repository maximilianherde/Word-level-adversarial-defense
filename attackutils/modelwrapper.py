"""

Custom model wrappers for attacking all our models using TextAttack and PyTorch.

"""


import textattack
import torchtext
import torch
from transformers import BertTokenizer


class CustomPyTorchModelWrapper(textattack.models.wrappers.model_wrapper.ModelWrapper):
    """
    Model wrapper for recurrent models and CNNs. Not to be used with SEM embeddings.
    """

    def __init__(self, model, outdim, vocab, device, tokenizer=torchtext.data.utils.get_tokenizer("basic_english")):
        self.model = model
        self.tokenizer = tokenizer
        self.outdim = outdim
        self.vocab = vocab
        self.device = device

    def __call__(self, text_input_list):
        preds = torch.zeros(size=(len(text_input_list), self.outdim))
        for i, review in enumerate(text_input_list):
            tokens = self.tokenizer(review)
            input = self.vocab.get_vecs_by_tokens(tokens)
            with torch.no_grad():
                prediction = self.model(
                    torch.unsqueeze(input, dim=0).to(self.device))
                preds[i] = prediction

        return preds


class CustomBERTModelWrapper(textattack.models.wrappers.model_wrapper.ModelWrapper):
    """
    Model wrapper for BERT models.
    """

    def __init__(self, model, outdim, device, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)):
        self.model = model
        self.tokenizer = tokenizer
        self.outdim = outdim
        self.device = device

    def __call__(self, text_input_list):
        preds = torch.zeros(size=(len(text_input_list), self.outdim))
        for i, review in enumerate(text_input_list):
            dict_ = self.tokenizer(
                review, padding="max_length", return_tensors='pt', max_length=512, truncation=True)
            with torch.no_grad():
                prediction = self.model(dict_["input_ids"].to(self.device), dict_[
                                        "token_type_ids"].to(self.device), dict_["attention_mask"].to(self.device))
                preds[i] = prediction

        return preds


class CustomSEMModelWrapper(textattack.models.wrappers.model_wrapper.ModelWrapper):
    """
    Model wrapper to be used with SEM embeddings.
    """

    def __init__(self, model, outdim, vocab, device, tokenizer=torchtext.data.utils.get_tokenizer("basic_english")):
        self.model = model
        self.tokenizer = tokenizer
        self.outdim = outdim
        self.vocab = vocab
        self.device = device

    def __call__(self, text_input_list):
        preds = torch.zeros(size=(len(text_input_list), self.outdim))
        for i, review in enumerate(text_input_list):
            review = review.lower()
            tokens = self.tokenizer(review)
            input_list = []
            for _token in tokens:
                embed_temp = self.vocab.get(_token)
                if embed_temp != None:
                    input_list.append(self.vocab.get(_token))

                else:
                    input_list.append(torch.zeros(50))

            input_stacked = torch.stack(input_list)
            with torch.no_grad():
                prediction = self.model(torch.unsqueeze(
                    input_stacked, dim=0).to(self.device))
                preds[i] = prediction

        return preds
