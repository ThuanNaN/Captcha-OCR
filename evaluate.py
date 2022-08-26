import os 

from sklearn.metrics import accuracy_score

from utils import *
from dataset import *
from CRNN import SimpleOCR, load_checkpoint

import torch
from torch.utils.data import  DataLoader

device = ('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = "./captcha_images_v2"
images = sorted(list(map(str, list(Path(data_dir).glob("*.png")))))

chars = get_chars()
vocab = get_vocab(chars)

_, test_sampler = train_val_split(images)
test_set = Captcha(data_dir,  sampler=test_sampler, binarize=True)

BATCH_SIZE = 32

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=0)

PATH = "./checkpoint/model.pt"
checkpoint = load_checkpoint(PATH)

model = SimpleOCR(in_channels=3, num_outputs=len(vocab), IMAGE_SHAPE=checkpoint["IMAGE_SHAPE"]).to(device)

model.load_state_dict(checkpoint["model_state_dict"])


preds = []
targets = []

model.eval()
with torch.no_grad():
    for inputs, labels in test_set:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs.unsqueeze(0)).permute(1, 0, 2)

        outputs = outputs.log_softmax(2).argmax(2)
        
        pred = ''
        then = 0
        for x in outputs[0]:
            if then != x:
                if x > 0 :
                    pred += int2label([x], vocab)
            then = x

        preds.append(pred)
        targets.append(int2label(labels.cpu().detach().tolist(), vocab))



assert len(preds) == len(targets)
cer = CER(preds, targets)

print("CER: ", cer())

acc = accuracy_score(targets, preds)
print("Accarucy score: ", acc)
