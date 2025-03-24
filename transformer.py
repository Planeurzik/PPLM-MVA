import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import Dataset
from models import LanguageModel

batch_size = 2
n_ctx = 1000
dataset = Dataset("dataset/nice_output2.txt", batch_size,n_ctx, "bpe_tokenizer.json")
n_token = 5000

n_embed = 256
model = LanguageModel(n_head = 6, 
                      head_size = 16,
                      head_output_dim = 16,
                      n_embed = n_embed,
                      n_hidden = 4 * n_embed,
                      n_layer = 4,
                      n_token = n_token,
                      n_ctx = n_ctx)

for i in range(100):
    tokens = next(dataset).to(device)

quit()

@torch.no_grad()
def estimate_loss(model, eval_iters = 100):
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


def train(model, n_iterations = 10000, learning_rate = 1e-3, eval_interval = 1000, eval_iters = 100):
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(n_iterations):
        # every once in a while evaluate the loss on train and validation sets
        if iter % eval_interval == 0 or iter == n_iterations - 1:
            losses = estimate_loss(model, eval_iters)
            print(f"step {iter}: train loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")

        X,Y = get_batch("train")
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

print(sum(p.numel() for p in model.parameters())/1e6, ' M parameters')
train(model, n_iterations = 20000, learning_rate = 1e-3, eval_interval = 1000, eval_iters = 100)

