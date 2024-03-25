# GPT research

This repo is mainly my hand-on experiments based on Andrej Karpathy tutorial videos.

## Installation

```
pip install -e .
```

## Usage

This python package is called `tinygpt` and use cli to perform train, val, and generate. Checkout `exps/train.sh` for more.

Example Output:

```
Namespace(name='TinyGPT', project='exps', task='train', data_path='../dataset/shakespeare.txt', batch_size=64, chunk_size=256, emb_size=128, seed=2202, learning_rate=0.001, train_iter=5000, val_iter=500, device='cuda', head_size=128, num_head=6, num_block=6, dropout=0.2, data_split=0.9)
Project path: exps/TinyGPT4
Reading data...

 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz 65
Train sample: 1003854 Val sample: 111540
Reading data...DONE
Building model...
3.202304 M parameters
Building model...DONE
Training...
  0%|          | 0/5000 [00:00<?, ?it/s]
Saved best iter 0 vloss 3.8561063 exps/TinyGPT4/best.pt
iter [    0/ 5000] tloss 3.844327926635742 vloss 3.8561062812805176:  10%|▉         | 499/5000 [00:24<03:14, 23.17it/s]
Saved best iter 500 vloss 2.1297767 exps/TinyGPT4/best.pt
iter [  500/ 5000] tloss 2.0727438926696777 vloss 2.1297767162323:  20%|██        | 1000/5000 [00:48<02:53, 23.09it/s] 
Saved best iter 1000 vloss 1.7895925 exps/TinyGPT4/best.pt
iter [ 1000/ 5000] tloss 1.6127829551696777 vloss 1.7895925045013428:  30%|██▉       | 1498/5000 [01:13<02:32, 23.00it/s]
Saved best iter 1500 vloss 1.6454413 exps/TinyGPT4/best.pt
iter [ 1500/ 5000] tloss 1.447810173034668 vloss 1.6454412937164307:  40%|███▉      | 1999/5000 [01:37<02:10, 22.96it/s] 
Saved best iter 2000 vloss 1.5828555 exps/TinyGPT4/best.pt
iter [ 2000/ 5000] tloss 1.3765604496002197 vloss 1.582855463027954:  50%|█████     | 2500/5000 [02:02<01:49, 22.92it/s]
Saved best iter 2500 vloss 1.5523161 exps/TinyGPT4/best.pt
iter [ 2500/ 5000] tloss 1.3373849391937256 vloss 1.5523160696029663:  60%|█████▉    | 2998/5000 [02:26<01:27, 22.90it/s]
Saved best iter 3000 vloss 1.5199503 exps/TinyGPT4/best.pt
iter [ 3000/ 5000] tloss 1.2968722581863403 vloss 1.519950270652771:  70%|██████▉   | 3499/5000 [02:51<01:05, 22.90it/s] 
Saved best iter 3500 vloss 1.5022203 exps/TinyGPT4/best.pt
iter [ 3500/ 5000] tloss 1.2683701515197754 vloss 1.5022202730178833:  80%|████████  | 4000/5000 [03:15<00:43, 22.89it/s]
Saved best iter 4000 vloss 1.4940672 exps/TinyGPT4/best.pt
iter [ 4000/ 5000] tloss 1.2467355728149414 vloss 1.4940671920776367:  90%|████████▉ | 4498/5000 [03:40<00:21, 22.89it/s]
Saved best iter 4500 vloss 1.4880433 exps/TinyGPT4/best.pt
iter [ 4500/ 5000] tloss 1.2330191135406494 vloss 1.4880433082580566: 100%|█████████▉| 4999/5000 [04:04<00:00, 22.88it/s]
Saved best iter 4999 vloss 1.4737064 exps/TinyGPT4/best.pt
iter [ 4999/ 5000] tloss 1.2203805446624756 vloss 1.4737063646316528: 100%|██████████| 5000/5000 [04:07<00:00, 20.19it/s]
Training...DONE
loss_train: 1.2203805 loss_val: 1.4737064

Anon's truth': not this next laws it did I:
But thinks hark discorrubjects, and then
Lear me; she will we die to be suit in: and the
dumb suffect I yearly; but, must to charge their
didle-wax; therefore lefinds thus a pack old mine sout this
renown more than than 'bway I have elord'd mine.'
Mom at Iwin France is not our ming; if
Thou arain cansquaged with law: that it is not dead,
As we'lt gold pardering with him grace;
That you, 'gays, envited the kitiest that you have,
And none to help to mell
```

## TODO

- Validate, and generate CLI
- Model pretrain on OWT dataset
- GPT-2 weight
- Finetune GPT-2
- Llama2 research

## References

- Tutorial video: https://www.youtube.com/watch?v=kCc8FmEb1nY
- Tutorial repo: https://github.com/karpathy/ng-video-lecture
- Attention Is All You Need: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
