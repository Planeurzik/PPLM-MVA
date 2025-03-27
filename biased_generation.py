import torch
import os
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import Dataset, load_tokenizer
from models import LanguageModel
import time

batch_size = 8
n_ctx = 200
tokenizer_path = "bpe_tokenizer.json"
tokenizer = load_tokenizer(tokenizer_path)
train_dataset = Dataset("dataset/train.txt", batch_size, n_ctx, tokenizer)
test_dataset = Dataset("dataset/test.txt", batch_size, n_ctx, tokenizer)
n_token = 10000
save_path = "checkpoints/checkpoint_tiny.pt"
#load_path = "checkpoints/checkpoint_huge.pt"
#load_path = "checkpoints/checkpoints_toeplitz.pt"
load_path ="checkpoints/checkpoint_tiny.pt"
TRAIN  = True

lr = 5e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_embed = 512
model = LanguageModel(n_head = 4, 
                      head_size = 64,
                      head_output_dim = 64,
                      n_embed = n_embed,
                      n_hidden = 4 * n_embed,
                      n_layer = 5,
                      n_token = n_token,
                      n_ctx = n_ctx)

#model = nn.DataParallel(model)

model = model.to(device)

def generate_kv_biased(model, inp_tokens, pasx, device, n_tok_max = 200, T=1, kv_cache= None):
    i = len(inp_tokens)
    tokens = torch.tensor(inp_tokens).to(device)
    tokens = torch.nn.functional.pad(tokens,(0,n_tok_max-tokens.shape[0]))
    logits, loss, kv_cache =model(tokens[None,:i], kv_cache = kv_cache)
    logits = logits[0,i-1]
    with torch.no_grad():
        while i<n_tok_max:
            logits_i = logits/T
            top_k_values, top_k_indices = torch.topk(logits_i, 50)
            top_k_probs = F.softmax(top_k_values,0)
            token_k_id = torch.multinomial(top_k_probs, num_samples=1)
            token_id = top_k_indices[token_k_id[0]]
            tokens[i] = token_id
            logits, loss, kv_cache =model(tokens[None,i:i+1], kv_cache = kv_cache)
            logits = logits[0,0]
            log_probs = pasx(logits)
            print(log_probs)
            i+=1
    return tokens, kv_cache

def create_bag_of_words(folder_path):
    bag_of_words = {}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and filename.endswith('.txt'):
            topic = os.path.splitext(filename)[0]
            with open(file_path, 'r', encoding='utf-8') as file:
                words = file.read().splitlines()
                bag_of_words[topic] = words
    
    return bag_of_words

class BoWAttributeModel(nn.Module):
    def __init__(self, topics_dict, tokenizer, vocab_size=10000):
        """
        Initialize the BoWAttributeModel with a dictionary of topics.

        :param topics_dict: Dictionary where keys are topics and values are lists of words.
        :param tokenizer: Tokenizer to convert words to token IDs.
        :param vocab_size: Size of the vocabulary.
        """
        super().__init__()
        self.topics_dict = topics_dict
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.topic_masks = self._build_topic_masks()

    def _build_topic_masks(self):
        """
        Build a mask for each topic.
        Each mask is a tensor of shape (vocab_size,) with 1.0 for token IDs that belong to the topic and 0.0 elsewhere.
        """
        topic_masks = {}
        for topic, words in self.topics_dict.items():
            mask = torch.zeros(self.vocab_size)
            for word in words:
                # Tokenize the word without adding special tokens
                token_ids = self.tokenizer.encode(word, add_special_tokens=False)
                # Tokenize with space prefix (handles cases like 'Ġword')
                token_ids_with_space = self.tokenizer.encode(" " + word, add_special_tokens=False)
                # In case the word is tokenized into multiple tokens, we mark all of them.
                for token_id in token_ids:
                    if token_id < self.vocab_size:  # Safety check
                        mask[token_id] = 1.0
            topic_masks[topic] = mask
        return topic_masks

    def get_log_topic_prob(self,logits,topic):
        mask =self.topic_masks[topic].to(logits.device)
        topic_sum = torch.sum(logits * mask.view(1, 1, -1), dim=-1)
        return topic_sum



    def forward(self, logits, topic = 0):
        """
        Compute the log probability that the generated tokens (given by logits) belong to each topic.

        :param logits: Tensor of shape (batch_size, vocab_size)
        :return: Dictionary mapping each topic to a tensor of log probabilities of shape (batch_size,)
        """
        # Convert logits to probabilities.
        #probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, nb_tokens_sentence, vocab_size)
        log_topic_probs = {}
        
        for topic in range(len(self.topic_masks)):
            log_topic_probs[topic] = self.get_log_topic_prob(logits, topic)
        return log_topic_probs


folder_path = 'wordlists'
topic_bow = create_bag_of_words(folder_path)
print("Bag of Words:", topic_bow)
bag_of_words_model = BoWAttributeModel(topic_bow,tokenizer).to(device)

print(sum(p.numel() for p in model.parameters())/1e6, ' M parameters')
checkpoint = torch.load(load_path)
print("loading checkpoint epoch ",checkpoint["epoch"])
model.load_state_dict(checkpoint['model_state_dict'])
text = "Happy Happy good good fun great"
text_tokens = tokenizer.encode(text)
generate_kv_biased(model,text_tokens,bag_of_words_model, device, n_tok_max=100)
exit()
for int_text_long in next(iter(train_dataset)):
    print("-----------------------------------------------")
    int_tokens = int_text_long[:n_ctx//2]
    out_tokens, kv_cache = generate_kv_biased(model,int_tokens, bag_of_words_model, device, n_tok_max = n_ctx)
    out_tokens = list(out_tokens)
    out_tokens.insert(n_ctx//2, 0)
    out_tokens.insert(n_ctx//2, 0)
    out_tokens.insert(n_ctx//2, 0)
    print(tokenizer.decode(out_tokens))


