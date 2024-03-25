import math

import torch
from torch import nn
from torch.nn import functional as F

class Head(nn.Module):

    def __init__(self, cfgs):
        super().__init__()

        self.cfgs = cfgs

        self.query = nn.Linear(self.cfgs.emb_size, self.cfgs.head_size, bias=False)
        self.key = nn.Linear(self.cfgs.emb_size, self.cfgs.head_size, bias=False)
        self.value = nn.Linear(self.cfgs.emb_size, self.cfgs.head_size, bias=False)
        self.dropout = nn.Dropout(self.cfgs.dropout)
        self.dropout2 = nn.Dropout(self.cfgs.dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer('tril', torch.tril(torch.ones(self.cfgs.chunk_size, self.cfgs.chunk_size)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x) # B, T, C
        k = self.key(x)   # B, T, C
        v = self.value(x) # B, T, C

        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.cfgs.dropout if self.training else 0, is_causal=True)
        else:
            wei = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1]) # (B, T, C) x (B, C, T) -> B, T, T
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v # (B, T, T) x (B, T, C) -> (B, T, C)

        out = self.dropout2(out)

        return out
    
class MultiHead(nn.Module):

    def __init__(self, cfgs):
        super().__init__()

        self.cfgs = cfgs

        self.heads = nn.ModuleList([Head(cfgs) for _ in range(self.cfgs.num_head)])
        self.project = nn.Linear(self.cfgs.head_size*self.cfgs.num_head, self.cfgs.emb_size, bias=False)
        self.dropout = nn.Dropout(self.cfgs.dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.project(x)
        x = self.dropout(x)
        return x
    
class FeedForward(nn.Module):

    def __init__(self, cfgs):
        super().__init__()

        self.cfgs = cfgs

        ACT = getattr(torch.nn, self.cfgs.activation)

        self.feed = nn.Sequential(
            nn.Linear(self.cfgs.emb_size, self.cfgs.emb_size*4),
            ACT(),
            nn.Linear(self.cfgs.emb_size*4, self.cfgs.emb_size),
            nn.Dropout(self.cfgs.dropout)
        )

    def forward(self, x):
        return self.feed(x)
    
class MultiHeadBlock(nn.Module):

    def __init__(self, cfgs):
        super().__init__()

        self.cfgs = cfgs

        self.heads = MultiHead(cfgs)
        self.feed = FeedForward(cfgs)

        self.head_norm = nn.LayerNorm(self.cfgs.emb_size)
        self.feed_norm = nn.LayerNorm(self.cfgs.emb_size)

    def forward(self, x):
        x = x + self.heads(self.head_norm(x))
        x = x + self.feed(self.feed_norm(x))
        return x
    
class GPTLanguageModel(nn.Module):

    def __init__(self, cfgs):
        super().__init__()

        self.cfgs = cfgs

        self.token_embedding_table = nn.Embedding(self.cfgs.corpus_size, self.cfgs.emb_size)
        self.position_embedding_table = nn.Embedding(self.cfgs.chunk_size, self.cfgs.emb_size)

        self.blocks = nn.Sequential(*[MultiHeadBlock(cfgs) for _ in range(self.cfgs.num_block)])
        self.lm_heads = nn.Linear(self.cfgs.emb_size, self.cfgs.corpus_size, bias=False)

        self.ln = nn.LayerNorm(self.cfgs.emb_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None): # B, T
        B, T = idx.shape

        pos_pos = self.position_embedding_table(torch.arange(T, device=self.cfgs.device)) # (T,C)
        tok_emb = self.token_embedding_table(idx) # B, T, C

        x = tok_emb + pos_pos
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_heads(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_c = logits.view(B*T, C)
            target_c = target.view(B*T)
            loss = F.cross_entropy(logits_c, target_c)

        return logits, loss
    
    def generate(self, x, max_new_token=500):

        for _ in range(max_new_token):
            logits, _ = self(x[:, -self.cfgs.chunk_size:])

            logits = logits[:, -1, :] # focus on last prediction, B, C
            logits = F.softmax(logits, dim=-1) # get probabilities, B, C
            next_logit = torch.multinomial(logits, 1) # B, 1
            
            x = torch.cat([x, next_logit], dim=1) # B, T + 1

        return x