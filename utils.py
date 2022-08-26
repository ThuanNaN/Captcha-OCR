

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import utils
from torch.utils.data import SubsetRandomSampler



def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        try:
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
            num_batches += 1
        except Exception as e:
            print(e)
        
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def train_val_split(dataset, val_split=0.2, random_seed=42, shuffle=False):
    indices = list(range(len(dataset)))
    split = int(val_split * len(dataset))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler


def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0]) 
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def label2int(label, vocab):
    int_label = []
    for c in label:
        if c in vocab:
            int_label.append(vocab[c])
        else:
            int_label.append(0) 
    return int_label

def int2label(int_label, vocab):
    label = ""
    for i in int_label:
        for k, v in vocab.items():
            if i == v:
                label += k
    return label

def ctc_loss(x, targets):
	bs = x.shape[1]
	log_probs = F.log_softmax(x, 2)
 
	input_lengths = torch.full(
	        size=(bs, ), fill_value=log_probs.size(1), dtype=torch.int32
	)
 
	target_lengths = torch.full(
	        size=(bs,), fill_value=targets.size(1), dtype=torch.int32
	)
 
	loss = nn.CTCLoss(blank=0)(
	        log_probs, targets, input_lengths, target_lengths
	)
 
	return loss

def Levenshtein_distance(pred_tokens, target_tokens):
        dp = [[0] * (len(target_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
        for i in range(len(pred_tokens) + 1):
            dp[i][0] = i
        for j in range(len(target_tokens) + 1):
            dp[0][j] = j
        for i in range(1, len(pred_tokens) + 1):
            for j in range(1, len(target_tokens) + 1):
                if pred_tokens[i - 1] == target_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        return dp[-1][-1]

class CER():
    def __init__(self, preds, targets):
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(targets, str):
            targets = [targets]

        errors = torch.tensor(0, dtype=torch.float)
        total = torch.tensor(0, dtype=torch.float)
        for pred, tgt in zip(preds, targets):
            pred_tokens = pred
            tgt_tokens = tgt
            errors += Levenshtein_distance(list(pred_tokens), list(tgt_tokens))
            total += len(tgt_tokens)

        self.errors = errors
        self.total = total
        
    def __call__(self) -> torch.Tensor:
        return self.errors / self.total

def show_loss(loss_history):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=loss_history)
    plt.title("Unique characters - Non-binarized")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()