import tqdm
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tinygpt import TQDM_BAR_FORMAT
from tinygpt.cfg import get_cfg, yaml_save, cfg2dict
from tinygpt.model import GPTLanguageModel
from tinygpt.dataset import Dataset
from tinygpt.utils import increment_path, init_seeds

@torch.no_grad()
def perform_validate(model, dataset, num_sample=100):
    model.eval()

    losses = []
    for split in ["train", "val"]:
        loss_split = 0
        for _ in range(num_sample):
            xs, ys = dataset.get_batch(split)
            _, loss = model(xs, ys)
            loss_split += loss
        loss_split /= num_sample
        losses.append(loss_split)

    model.train()

    return losses[0], losses[1]

def train_sequence(cfgs):

    save_dir = Path(increment_path(Path(cfgs.project) / cfgs.name, exist_ok=cfgs.task != "train", mkdir=cfgs.task == "train"))
    save_last, save_best = save_dir / 'last.pt', save_dir / 'best.pt'  # checkpoint paths
    save_config = save_dir / 'config.yaml'
    if cfgs.task == "train":
        yaml_save(save_config, cfg2dict(cfgs))
    print(f"Project path: {save_dir}")

    print("Reading data...")
    dataset = Dataset(cfgs)
    print("Reading data...DONE")

    cfgs.corpus_size = dataset.corpus_size

    print("Building model...")
    model = GPTLanguageModel(cfgs).to(cfgs.device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    print("Building model...DONE")

    print("Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfgs.learning_rate)

    best_vloss = 9999
    pbar = tqdm.tqdm(list(range(cfgs.train_iter)), total=cfgs.train_iter, bar_format=TQDM_BAR_FORMAT)
    des = ""
    for iter in pbar:

        xs, ys = dataset.get_batch("train")
        _, loss = model(xs, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter % cfgs.val_iter == 0) or (iter == cfgs.train_iter-1):
            loss_train, loss_val = perform_validate(model, dataset)

            if loss_val < best_vloss:
                torch.save(model.state_dict(), save_best)
                best_vloss = loss_val
                print("\nSaved best iter", iter, "vloss", best_vloss.cpu().detach().numpy(), save_best)

            torch.save(model.state_dict(), save_last)
            des = f"iter [{iter:5d}/{cfgs.train_iter:5d}] tloss {loss_train.cpu().detach().numpy()} vloss {loss_val.cpu().detach().numpy()}"
            
        pbar.set_description(des)

    print("Training...DONE")

    print("loss_train:", loss_train.cpu().detach().numpy(), "loss_val:", loss_val.cpu().detach().numpy())
    print(dataset.decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=cfgs.device), max_new_token=500)[0].tolist()), "\n")
