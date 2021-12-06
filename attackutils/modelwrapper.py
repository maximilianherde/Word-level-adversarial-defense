import textattack
import torchtext
import torch

class CustomPyTorchModelWrapper(textattack.models.wrappers.model_wrapper.ModelWrapper):
    def __init__(self, model, outdim, vocab=torchtext.vocab.GloVe("6B", dim=50), tokenizer=torchtext.data.utils.get_tokenizer("basic_english")):
        self.model = model
        self.tokenizer = tokenizer
        self.outdim = outdim
        self.vocab = vocab

    def __call__(self, text_input_list):
        preds = torch.zeros(size=(len(text_input_list), self.outdim))
        for i, review in enumerate(text_input_list):
            tokens = self.tokenizer(review)
            input = self.vocab.get_vecs_by_tokens(tokens)
        with torch.no_grad():
            prediction = self.model(torch.unsqueeze(input, dim=0))
            preds[i] = prediction

        return preds
