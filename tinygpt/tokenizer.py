import json

MAX_BYTE = 256

def get_stats(data):
    stats = {}
    for pair in zip(data[:-1], data[1:]):
        stats[pair] = stats.get(pair, 0) + 1
    
    pair_stats = [(f, p) for p, f in stats.items()]
    pair_stats = sorted(pair_stats, reverse=True) 

    return pair_stats

def merge(data, pair, pair_idx):
    
    new_data = []
    idx = 0
    while True:
        # check if pair is pair
        if data[idx:idx+2] == list(pair):
            new_data.append(pair_idx)
            idx += 2
        else:
            new_data.append(data[idx])
            idx += 1
        
        if idx >= len(data):
            break

    return new_data

class Tokenizer:

    def __init__(self, token_path):
        self._token_path = token_path
        self.load(self._token_path)

        print("Tokenizer initialized from:", self._token_path)
        print("Tokenizer corpus size:", self.corpus_size)

    @property
    def corpus_size(self):
        return MAX_BYTE+len(self.ids_map)

    def _encode(self, raw_idx: list[int]) -> list[int]:
        target_idx = list(raw_idx)

        for idx, pair in enumerate(self.ids_map):
            target_idx = merge(target_idx, pair, idx+MAX_BYTE)

        return target_idx

    def encode(self, text: str) -> list[int]:
        raw_bytes = text.encode('utf-8')
        raw_bytes = list(map(lambda x: int(x), raw_bytes))
        return self._encode(raw_bytes)

    def _decode(self, raw_idx: list[int]) -> list[int]:
        target_idx = list(raw_idx)
        for idx, pair in reversed(list(enumerate(self.ids_map))):
            new_idx = []
            for i in target_idx:
                if i == idx+MAX_BYTE:
                    new_idx.extend(pair)
                else:
                    new_idx.append(i)
            target_idx = new_idx
        
        return target_idx

    def decode(self, raw_idx: list[int]) -> str:
        decoded_idx = self._decode(raw_idx)
        decoded_bytes = b"".join([bytes([b]) for b in decoded_idx])
        return decoded_bytes.decode("utf-8")

    def train(self, raw_idx, num_iter=100):
        ids_map = []
        compress_curve = []

        target_idx = list(raw_idx) # copy
        for iter in range(num_iter):
            pair_stats = get_stats(target_idx)
            top_pair_freg, top_pair = pair_stats[0]
            if top_pair_freg <= 1:
                break

            target_idx = merge(target_idx, top_pair, len(ids_map)+MAX_BYTE)
            ids_map.append(top_pair)

            compress_curve.append(len(raw_idx)/len(target_idx))

        print("stop at iter", iter+1, "/", num_iter)
        print("number of ids_map", len(ids_map))
        print("initial data size:", len(raw_idx))
        print("final data size:", len(target_idx))
        print("size difference:", len(raw_idx)-len(target_idx))
        print("compress ratio:", len(raw_idx)/len(target_idx))

        # target_idx, compress_curve
        self.ids_map = ids_map
    
    def save(self, target_path):
        with open(target_path, "w") as f:
            json.dump(self.ids_map, f)

    def load(self, target_path):
        with open(target_path, "r") as f:
            self.ids_map = json.load(f)

        # convert pair to tuple
        self.ids_map = [(a, b) for a, b in self.ids_map]

        