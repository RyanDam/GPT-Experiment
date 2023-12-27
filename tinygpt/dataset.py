import torch

class Dataset:

    def __init__(self, cfgs):

        self.cfgs = cfgs
        
        with open(cfgs.data_path, 'r') as f:
            self.data_raw = f.read()

        self.corpus = sorted(list(set(self.data_raw)))
        self.corpus_size = len(self.corpus)
        print("".join(self.corpus), len(self.corpus))

        self.ctoi = {c: idx for idx, c in enumerate(self.corpus)}
        self.itoc = {idx: c for idx, c in enumerate(self.corpus)}
        self.encode = lambda s: [self.ctoi[c] for c in s]
        self.decode = lambda tokens: "".join([self.itoc[i] for i in tokens])

        self.data_token = torch.tensor(self.encode(self.data_raw)).long().to(cfgs.device)

        self.data_length = len(self.data_token)
        self.data_split_idx = int(self.cfgs.data_split*self.data_length)
        self.data_token_train = self.data_token[:self.data_split_idx]
        self.data_token_val = self.data_token[self.data_split_idx:]
        print("Train sample:", len(self.data_token_train), "Val sample:", len(self.data_token_val))

    def get_batch(self, split):
        data_split = self.data_token_train if split == "train" else self.data_token_val
        max_samples = len(data_split)

        idxes = torch.randint(max_samples - self.cfgs.chunk_size - 1, (self.cfgs.batch_size,))
        xs = torch.stack([data_split[idx:idx+self.cfgs.chunk_size] for idx in idxes])
        ys = torch.stack([data_split[idx+1:idx+self.cfgs.chunk_size+1] for idx in idxes])
        return xs.to(self.cfgs.device), ys.to(self.cfgs.device)
    