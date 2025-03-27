import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import Dataset, load_tokenizer
from models import LanguageModel
import time

batch_size = 8
n_ctx = 500
tokenizer_path = "bpe_tokenizer.json"
tokenizer = load_tokenizer(tokenizer_path)
train_dataset = Dataset("dataset/trainb.txt", batch_size, n_ctx, tokenizer)
test_dataset = Dataset("dataset/testb.txt", batch_size, n_ctx, tokenizer)
n_token = 10000
save_path = "checkpoints/checkpoint_huge.pt"
#load_path = "checkpoints/checkpoint_huge.pt"
#load_path = "checkpoints/checkpoints_toeplitz.pt"
load_path = None
TRAIN  = True

lr = 5e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_embed = 1048
model = LanguageModel(n_head = 16, 
                      head_size =64,
                      head_output_dim =64,
                      n_embed = n_embed,
                      n_hidden = 4 * n_embed,
                      n_layer = 12,
                      n_token = n_token,
                      n_ctx = n_ctx)

#model = nn.DataParallel(model)

model = model.to(device)

@torch.no_grad()
def estimate_loss(model):
    ptime = time.time()
    losses = []
    for batch in test_dataset:
        batch = batch.to(device)
<<<<<<< HEAD
        logits, loss, kv_cache = model(batch)
=======
        logits, loss = model(batch)
        #loss = torch.sum(loss)
>>>>>>> refs/remotes/origin/main
        losses.append(loss.item())
    print(time.time()-ptime)
    return np.mean(losses)


def train(model, epochs = 10000, learning_rate = 3e-4, eval_interval = 1000, save_path="checkpoints.pt"):
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    losses_train = []
    losses_test = []

    loss_mean=0
    i=0
    for epoch in range(epochs):
        k=0
        for batch in train_dataset:
            i+=1
            batch = batch.to(device)
            _, loss, kv_cache = model(batch)
            #loss = torch.sum(loss)
            loss_at_step = loss.item()
            loss_mean+=loss_at_step
            if k % eval_interval == 0:
                loss_test = estimate_loss(model)
                loss_mean = loss_mean/i
                print(f"Epoch {epoch}, step {k}: train loss {loss_mean:.4f}, test loss {loss_test:.4f}")
                checkpoint = {
                    'epoch': epoch,
                    'step': k,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': loss_at_step,
                    'test_loss': loss_test
                }
                torch.save(checkpoint, save_path)
                print(f"Model checkpoint saved at {save_path}")
                losses_train.append(loss_at_step)
                losses_test.append(loss_test)

                out_string = inference(model, tokenizer, int_text, n_tok_max =n_ctx)
                print(out_string)

                loss_mean=0
                i=0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            k+=1
    return losses_train, losses_test

def inference(model, tokenizer, tokens, n_tok_max = 100):
    #tokens = tokenizer.encode(start_string)
    n_tokens, kv_cache = model.generate_kv(tokens, device, n_tok_max =n_tok_max)
    #n_tokens = model.generate(tokens, device, n_tok_max =n_tok_max)
    string = tokenizer.decode(n_tokens)
    return string

print(sum(p.numel() for p in model.parameters())/1e6, ' M parameters')
if load_path is not None:
    checkpoint = torch.load(load_path)
    print("loading checkpoint epoch ",checkpoint["epoch"])
    model.load_state_dict(checkpoint['model_state_dict'])

int_text = next(iter(train_dataset))[0][:n_ctx//2]
if TRAIN:
    losses_train, losses_test = train(model, epochs = 40, learning_rate =lr, eval_interval = 1000, save_path = save_path)

print(int_text)
out_string = inference(model, tokenizer, int_text, n_tok_max = n_ctx)
print(out_string)
