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
    def __init__(self, path, batch_size, n_ctx, tokenizer, n_buffer = 10):
        self.path = path
        self.tokenizer = tokenizer
        file =  open(self.path, "r")
        self.tokens = deque()
        self.batch_size = batch_size
        self.n_ctx=n_ctx
        self.n_buffer = n_buffer
        # To get the total number of lines in the file
        self.file = file
        
    def fill_tokens(self, n_tokens):

        while(len(self.tokens)<n_tokens):
            line = self.file.readline()
            if not line:
                raise StopIteration
            self.tokens.extend(self.tokenizer.encode(clean_text(line)))

    def get_batch(self):
        end = self.fill_tokens(self.batch_size*self.n_ctx*self.n_buffer)
        batch = [[self.tokens.popleft() for i in range(self.n_ctx)] for i in range(self.batch_size)]
        batch = torch.tensor(batch, dtype=torch.int)
        return batch

    def __next__(self):
        batch = self.get_batch()
        return batch

    def __iter__(self):
        self.file.seek(0)
        return self

def create_toeplitz_matrix_from_vector(vector):
    # Determine the size of the Toeplitz matrix
    n = vector.shape[0]

    # Create a range tensor for indexing
    row_indices = torch.arange(n, device= vector.device).unsqueeze(0)
    col_indices = torch.arange(n, device = vector.device).unsqueeze(1)

    # Use broadcasting to create the Toeplitz matrix
    toeplitz_matrix = vector[torch.abs(row_indices - col_indices)]

    return toeplitz_matrix

def proper_decode(tokens):
    return "".join(tokens).replace("Ġ", " ").strip()