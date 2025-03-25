import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np

class Head(nn.Module):
    def __init__(self, head_input_dim, head_size, head_output_dim, context_length):
        super().__init__()
        self.key = nn.Linear(head_input_dim, head_size, bias=False)
        self.query = nn.Linear(head_input_dim, head_size, bias=False)
        self.value = nn.Linear(head_input_dim, head_output_dim, bias=False)
        # Some Pytorch way of defining a matrix without trainable parameters 
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))     
        self.posbias = nn.Parameter(torch.zeros(context_length,context_length))
        self.context_length = context_length
        
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        # if training: B = batch_size, else B = 1
        # T = context_length
        # I = head_input_dim
        # H = head_size
        # O = head_output_dim
        
        k = self.key(x)   # (B, T, H)
        q = self.query(x) # (B, T, H)
        v = self.value(x) # (B, T, O)
        attention_scores = q @ k.transpose(1,2) # (B, T, H) @ (B, H, T) -> (B, T, T)
        attention_scores = attention_scores#+self.posbias[None,:,:]
        mask = torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1)
        masked_attention_scores = attention_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        attention_weights = torch.softmax(masked_attention_scores * self.head_size**-0.5, dim=-1) # (B, T, T)
        context_vectors = attention_weights @ v # (B, T, T) @ (B, T, O) -> (B, T, O)
        return context_vectors

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_head, head_size, head_output_dim, context_length, n_embed):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_input_dim = n_embed, 
                                         head_size = head_size, 
                                         head_output_dim = head_output_dim,
                                         context_length = context_length) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_output_dim, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True) # batch mean
        xvar = x.var(1, keepdim=True) # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_hidden, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_embed),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_head, head_size, head_output_dim, n_hidden, context_length, n_embed, n_token):
        super().__init__()
        self.self_attention_heads = MultiHeadAttention(n_head = n_head,
                                                       head_size = head_size,
                                                       head_output_dim = head_output_dim,
                                                       context_length= context_length,
                                                       n_embed = n_embed)
        self.ffwd = FeedForward(n_hidden = n_hidden, n_embed = n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # here are skip connections, also called residual connections
        # they help training deep neural networks by adding a pathway to the input
        x = x + self.self_attention_heads(x)

        # normalization layer; recent implementations put them before self attention heads!
        x = self.ln1(x)
        
        # and again skip connections:
        x = x + self.ffwd(x)

        # and again normalization layer
        x = self.ln2(x)

        return x

class LanguageModel(nn.Module):

    def __init__(self, n_head, head_size, head_output_dim, n_embed, n_hidden, n_layer, n_token, n_ctx):
        super().__init__()
        self.token_embedding_table = nn.Embedding(n_token, n_embed)
        self.position_embedding_table = nn.Embedding(n_ctx, n_embed)
        self.blocks = nn.Sequential(*[Block(n_head, head_size, head_output_dim, n_hidden, context_length = n_ctx, n_embed = n_embed, n_token=n_token) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, n_token)
        self.n_ctx = n_ctx

    def forward(self, idx):
        B, T = idx.shape
        # I = head_input_dim = head_output_dim = n_embed

        tok_emb = self.token_embedding_table(idx) # (B, T, I)
        pos_emb = self.position_embedding_table(torch.arange(T, device = idx.device)) # (T, I)
        x = tok_emb + pos_emb # (B, T, I)
        x = self.blocks(x) # (B, T, I)
        x = self.ln_f(x) # (B, T, I)
        logits = self.lm_head(x) # (B, T, n_token)

        y = idx[:,1:]
        B, T, n_token = logits.shape
        logits_shifted = logits[:,:-1]
        logits_shifted = logits_shifted.reshape(B*(T-1), n_token)
        y = y.reshape(B*(T-1))
        loss = F.cross_entropy(logits_shifted, y.long())

        return logits, loss
    
    def generate(self, inp_tokens, device, n_tok_max = 200, T=1):
        i = len(inp_tokens)-40
        tokens = torch.tensor(inp_tokens).to(device)
        tokens = torch.nn.functional.pad(tokens,(0,self.n_ctx-tokens.shape[0]))
        with torch.no_grad():
            while i<n_tok_max:
                logits, loss = self(tokens[None,:])
                logits_i = logits[0,i-1]/T
                top_k_values, top_k_indices = torch.topk(logits_i, 1000)
                top_k_probs = F.softmax(top_k_values)
                token_k_id = torch.multinomial(top_k_probs, num_samples=1)
                token_id = top_k_indices[token_k_id[0]]
                tokens[i] = token_id
                i+=1

            
        return tokens
