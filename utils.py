import torch
from collections import deque
import re
from transformers import PreTrainedTokenizerFast

def clean_text(text):
    # Define allowed characters: English letters, numbers, punctuation, common symbols, and whitespace
    allowed_chars = r"[^a-zA-Z0-9.,!?;:'\"()\[\]{}\-+=_/\\*&^%$#@<>’| \n”“•]"

    # Remove unwanted characters while keeping spaces and newlines
    return re.sub(allowed_chars, "~", text)

def load_tokenizer(path):
    return PreTrainedTokenizerFast(tokenizer_file=path)

class Dataset:
    def __init__(self, path, batch_size, n_tokens, tokenizer_path, n_buffer = 100):
        self.path = path
        self.tokenizer = load_tokenizer(tokenizer_path)
        file =  open(self.path, "r")
        self.tokens = deque()
        self.batch_size = batch_size
        self.n_tokens = n_tokens
        self.n_buffer = n_buffer
        # To get the total number of lines in the file
        self.file = file
        
    def fill_tokens(self, n_tokens):
        while(len(self.tokens)<n_tokens):
            line = self.file.read()
            if not line:
                return False
            self.tokens.extend(self.tokenizer.encode(clean_text(line)))
        return True

    def get_batch(self):
        end = self.fill_tokens(self.batch_size*self.n_tokens*self.n_buffer)
        if end: # if end of file reached return False
            return False
        batch = [[self.tokens.pop() for i in range(self.n_tokens)] for i in range(self.batch_size)]
        batch = torch.Tensor(batch)
        return batch

    def __next__(self):
        batch = self.get_batch()
        if not batch:
            raise StopIteration
        return batch

    def __iter__(self):
        self.file.seek(0)
        return self
