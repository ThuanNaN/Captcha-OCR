import os, sys
from pathlib import Path
import logging
from tqdm import tqdm

from CRNN import save_checkpoint
from utils import *

from dataset import Captcha, get_vocab, get_chars
from CRNN import SimpleOCR

import torch
from torch import utils
from torch.utils.data import  DataLoader

logging.basicConfig(format="%(message)s", level=logging.INFO)
LOGGER = logging.getLogger('CS338-OCR-Captcha')

device = ('cuda' if torch.cuda.is_available() else 'cpu') 

def fit(model, dataloaders, criterion, optimizer, num_epochs=50, scheduler=None):
    LOGGER.info(f"{colorstr('Optimizer:')} {optimizer}")
    LOGGER.info(f"\n{colorstr('Loss:')} {criterion.__name__}")
    loss_history = []
    for epoch in range(num_epochs):
        LOGGER.info(colorstr(f'\nEpoch {epoch}/{num_epochs - 1}:'))

        for phase in ['train']:
            running_loss = 0.0

            if phase == 'train':
                LOGGER.info(colorstr('black', 'bold', '%20s' + '%15s' * 2) % 
                            ('Training:', 'gpu_mem', 'loss'))
                model.train()
            else:
                LOGGER.info(colorstr('black', 'bold', '\n%20s' + '%15s' * 2) % 
                            ('Validation:','gpu_mem', 'loss'))
                model.eval()
            
            with tqdm(dataloaders[phase],
                      total=len(dataloaders[phase]),
                      bar_format='{desc} {percentage:>7.0f}%|{bar:10}{r_bar}{bar:-10b}',
                      unit='batch') as _phase:

                for inputs, labels in _phase:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase =='train'):
                        if phase == 'train':
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                    running_loss += loss.item() / len(dataloaders[phase])

                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}GB'
                    desc = ('%35s' + '%15.6g' * 1) % (mem, running_loss)
                    _phase.set_description_str(desc)
                    
            loss_history.append(running_loss)

            if phase == 'val' and scheduler:
                scheduler.step(running_loss)
    return model, loss_history



data_dir = "./captcha_images_v2"
images = images = sorted(list(map(str, list(Path(data_dir).glob("*.png")))))
labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in images]
train_sampler, test_sampler = train_val_split(images)
train_set = Captcha(data_dir, sampler=train_sampler, binarize=True)
test_set = Captcha(data_dir,  sampler=test_sampler, binarize=True)


BATCH_SIZE = 32
IMAGE_SHAPE = train_set[0][0].shape

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=0)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0)

chars = get_chars()
vocab = get_vocab(chars)
model = SimpleOCR(in_channels=3, num_outputs=len(vocab), IMAGE_SHAPE=IMAGE_SHAPE).to(device)
N_EPOCHS = 2
LEARNING_RATE = 3e-4
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = ctc_loss

_, loss_history = fit(model, 
    {'train': train_loader},
    criterion,
    optimizer,
    N_EPOCHS,
)

save_checkpoint(model, IMAGE_SHAPE)
show_loss(loss_history)
