import random
import math
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(1337)

batch_size = 64
chunk_size = 256 # how wide in time
emb_size =  128
learning_rate = 1e-3
train_iter = 5000
val_iter = 500
device = "mps"
head_size = 128
num_head = 6
num_block = 6
dropout = 0.2

print("Reading data...")

data_file = "dataset/shakespeare.txt"
with open(data_file, 'r') as f:
    data_raw = f.read()

corpus = sorted(list(set(data_raw)))
corpus_size = len(corpus)
print("".join(corpus), len(corpus))

ctoi = {c: idx for idx, c in enumerate(corpus)}
itoc = {idx: c for idx, c in enumerate(corpus)}
encode = lambda s: [ctoi[c] for c in s]
decode = lambda tokens: "".join([itoc[i] for i in tokens])

data_token = torch.tensor(encode(data_raw)).long().to(device)

data_length = len(data_token)
data_split_idx = int(0.9*data_length)
data_token_train = data_token[:data_split_idx]
data_token_val = data_token[data_split_idx:]
len(data_token_train), len(data_token_val)

print("Reading data...DONE")
print("Train sample:", len(data_token_train), "Val sample:", len(data_token_val))

def get_batch(split):
    data_split = data_token_train if split == "train" else data_token_val
    max_samples = len(data_split)

    idxes = torch.randint(max_samples - chunk_size - 1, (batch_size,))
    xs = torch.stack([data_split[idx:idx+chunk_size] for idx in idxes])
    ys = torch.stack([data_split[idx+1:idx+chunk_size+1] for idx in idxes])
    return xs.to(device), ys.to(device)

@torch.no_grad()
def perform_validate(model, num_sample=100):
    model.eval()

    losses = []
    for split in ["train", "val"]:
        loss_split = 0
        for _ in range(num_sample):
            xs, ys = get_batch(split)
            _, loss = model(xs, ys)
            loss_split += loss
        loss_split /= num_sample
        losses.append(loss_split)


    model.train()

    return losses[0], losses[1]

class Head(nn.Module):

    def __init__(self):
        super().__init__()

        self.query = nn.Linear(emb_size, head_size, bias=False)
        self.key = nn.Linear(emb_size, head_size, bias=False)
        self.value = nn.Linear(emb_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer('tril', torch.tril(torch.ones(chunk_size, chunk_size)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x) # B, T, C
        k = self.key(x)   # B, T, C
        v = self.value(x) # B, T, C

        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout if self.training else 0, is_causal=True)
        else:
            wei = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1]) # (B, T, C) x (B, C, T) -> B, T, T
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            out = wei @ v # (B, T, T) x (B, T, C) -> (B, T, C)

        out = self.dropout2(out)

        return out

class MultiHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(num_head)])
        self.project = nn.Linear(head_size*num_head, emb_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.project(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.feed = nn.Sequential(
            nn.Linear(emb_size, emb_size*4),
            nn.ReLU(),
            nn.Linear(emb_size*4, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.feed(x)

class MultiHeadBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.heads = MultiHead()
        self.feed = FeedForward()

        self.head_norm = nn.LayerNorm(emb_size)
        self.feed_norm = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = x + self.heads(self.head_norm(x))
        x = x + self.feed(self.feed_norm(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(corpus_size, emb_size)
        self.position_embedding_table = nn.Embedding(chunk_size, emb_size)

        self.blocks = nn.Sequential(*[MultiHeadBlock() for _ in range(num_block)])
        self.lm_heads = nn.Linear(emb_size, corpus_size, bias=False)

        self.ln = nn.LayerNorm(emb_size)

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

        pos_pos = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
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
            logits, _ = self(x[:, -chunk_size:])

            logits = logits[:, -1, :] # focus on last prediction, B, C
            logits = F.softmax(logits, dim=-1) # get probabilities, B, C
            next_logit = torch.multinomial(logits, 1) # B, 1
            
            x = torch.cat([x, next_logit], dim=1) # B, T + 1

        return x

print("Building model...")
m = GPTLanguageModel().to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
print("Building model...DONE")

print("Training...")

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for iter in range(train_iter):

    xs, ys = get_batch("train")
    logit, loss = m(xs, ys)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % val_iter == 0:
        loss_train, loss_val = perform_validate(m)
        print("step:", iter, "loss_train:", loss_train, "loss_val:", loss_val)

print("Training...DONE")

print("loss_train:", loss_train, "loss_val:", loss_val)
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_token=500)[0].tolist()), "\n")
