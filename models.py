import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils import create_toeplitz_matrix_from_vector

class Head(nn.Module):
    def __init__(self, head_input_dim, head_size, head_output_dim, context_length):
        super().__init__()
        self.key = nn.Linear(head_input_dim, head_size, bias=False)
        self.query = nn.Linear(head_input_dim, head_size, bias=False)
        self.value = nn.Linear(head_input_dim, head_output_dim, bias=False)
        # Some Pytorch way of defining a matrix without trainable parameters 
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))     
        self.posbias = nn.Parameter(torch.zeros(context_length,context_length))
        #self.posbias = nn.Parameter(torch.zeros(context_length,))
        self.context_length = context_length
        
        self.head_size = head_size

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        # if training: B = batch_size, else B = 1
        # T = context_length
        # I = head_input_dim
        # H = head_size
        # O = head_output_dim
         
        if kv_cache is None:
            k = self.key(x)   # (B, T, H)
            q = self.query(x) # (B, T, H)
            v = self.value(x) # (B, T, O)
            kv_cache = (k,v)
            attention_scores = q @ k.transpose(1,2) # (B, T, H) @ (B, H, T) -> (B, T, T)
            #posbiasmat = create_toeplitz_matrix_from_vector(self.posbias)
            attention_scores = attention_scores#+self.posbias[None,:,:]
            mask = torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1)
            masked_attention_scores = attention_scores.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
            attention_weights = torch.softmax(masked_attention_scores * self.head_size**-0.5, dim=-1) # (B, T, T)
            context_vectors = attention_weights @ v # (B, T, T) @ (B, T, O) -> (B, T, O)
            return context_vectors, kv_cache

        else:
            kc = kv_cache[0] # (B,T,H)
            vc = kv_cache[1] # (B,T,H)
            k = self.key(x)   # (B, 1, H)
            q = self.query(x) # (B, 1, H)
            v = self.value(x) # (B, 1, O)
            kc = torch.cat([kc,k], dim=1)
            vc = torch.cat([vc,v], dim=1)
            attention_scores = q @ kc.transpose(1,2) 
            attention_weights = torch.softmax(attention_scores*self.head_size**(-0.5), dim=-1)
            context_vector = attention_weights @ vc  # (B,1,T) @ (B,T,O) -> (B,1,0)

            return context_vector, (kc,vc)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_head, head_size, head_output_dim, context_length, n_embed):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_input_dim = n_embed, 
                                         head_size = head_size, 
                                         head_output_dim = head_output_dim,
                                         context_length = context_length) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_output_dim, n_embed)

    def forward(self, x, kv_cache=None):
        headsout = []
        nkv_cache = []
        for i, head in enumerate(self.heads):
            if kv_cache is not None:
                out, hkv_cache = head(x,kv_cache=kv_cache[i])
            else :
                out, hkv_cache = head(x,kv_cache=None)
            headsout.append(out)
            nkv_cache.append(hkv_cache)
        
        out = torch.cat(headsout, dim=-1)
        out = self.proj(out)
        return out, nkv_cache

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

    def forward(self, x, kv_cache = None):
        # here are skip connections, also called residual connections
        # they help training deep neural networks by adding a pathway to the input
        out, kv_cache =self.self_attention_heads(x,kv_cache=kv_cache)
        x = x + out

        # normalization layer; recent implementations put them before self attention heads!
        
        # and again skip connections:
        x = x + self.ffwd(x)

        # and again normalization layer

        return x, kv_cache

class LanguageModel(nn.Module):

    def __init__(self, n_head, head_size, head_output_dim, n_embed, n_hidden, n_layer, n_token, n_ctx):
        super().__init__()
        self.token_embedding_table = nn.Embedding(n_token, n_embed)
        self.position_embedding_table = nn.Embedding(n_ctx, n_embed)
        self.blocks = nn.Sequential(*[Block(n_head, head_size, head_output_dim, n_hidden, context_length = n_ctx, n_embed = n_embed, n_token=n_token) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embed, n_token)
        self.n_ctx = n_ctx

    def forward(self, idx, kv_cache = None):
        B, T = idx.shape
        # I = head_input_dim = head_output_dim = n_embed

        tok_emb = self.token_embedding_table(idx) # (B, T, I)
        pos_emb = self.position_embedding_table(torch.arange(T, device = idx.device)) # (T, I)
        x = tok_emb + pos_emb # (B, T, I)
        nkv_cache = []
        for i, block in enumerate(self.blocks):
            if kv_cache is not None:
                x, bkv_cache = self.blocks[i](x, kv_cache = kv_cache[i]) 
            else : 
                x, bkv_cache = self.blocks[i](x, kv_cache=None) 
            nkv_cache.append(bkv_cache)
        logits = self.lm_head(x) # (B, T, n_token)

        y = idx[:,1:]
        B, T, n_token = logits.shape
        logits_shifted = logits[:,:-1]
        logits_shifted = logits_shifted.reshape(B*(T-1), n_token)
        y = y.reshape(B*(T-1))
        loss = F.cross_entropy(logits_shifted, y.long())

        return logits, loss, nkv_cache
     
    
    def generate_kv(self, inp_tokens, device, n_tok_max = 200, T=1, kv_cache= None):
        i = len(inp_tokens)
        tokens = torch.tensor(inp_tokens).to(device)
        tokens = torch.nn.functional.pad(tokens,(0,n_tok_max-tokens.shape[0]))
        logits, loss, kv_cache = self(tokens[None,:i], kv_cache = kv_cache)
        logits = logits[0,i-1]
        with torch.no_grad():
            while i<n_tok_max:
                logits_i = logits/T
                top_k_values, top_k_indices = torch.topk(logits_i, 50)
                top_k_probs = F.softmax(top_k_values,0)
                token_k_id = torch.multinomial(top_k_probs, num_samples=1)
                token_id = top_k_indices[token_k_id[0]]
                tokens[i] = token_id
                logits, loss, kv_cache = self(tokens[None,i:i+1], kv_cache = kv_cache)
                logits = logits[0,0]
                i+=1
        return tokens, kv_cache


    def generate(self, inp_tokens, device, n_tok_max = 200, T=1):
        i = len(inp_tokens)
        tokens = torch.tensor(inp_tokens).to(device)
        tokens = torch.nn.functional.pad(tokens,(0,self.n_ctx-tokens.shape[0]))
        with torch.no_grad():
            while i<n_tok_max:
                logits, loss, kv_cache = self(tokens[None,:])
                logits_i = logits[0,i-1]/T
                top_k_values, top_k_indices = torch.topk(logits_i, 50)
                top_k_probs = F.softmax(top_k_values)
                token_k_id = torch.multinomial(top_k_probs, num_samples=1)
                token_id = top_k_indices[token_k_id[0]]
                tokens[i] = token_id
                i+=1

            
        return tokens
