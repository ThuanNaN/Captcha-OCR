import torch
import torch.nn as nn

import os

class Bidirectional(nn.Module):
    def __init__(self, inp, hidden, out, num_layers=1, batch_first=False, dropout=0, lstm=True):
        super().__init__()
        if lstm:
            self.rnn = nn.LSTM(inp, hidden, bidirectional=True, 
                               num_layers=num_layers, 
                               dropout=dropout, 
                               batch_first=batch_first)
        else:
            self.rnn = nn.GRU(inp, hidden, bidirectional=True, 
                              num_layers=num_layers,
                              dropout=dropout, 
                              batch_first=batch_first)
        
        self.embedding = nn.Linear(hidden * 2, out)

    def forward(self, inputs):
        recurrent, _ = self.rnn(inputs)
        out = self.embedding(recurrent)
        return out

class SimpleOCR(nn.Module):
    def __init__(self, in_channels, num_outputs, IMAGE_SHAPE):
        super().__init__()
        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=(6, 3), padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(4, 3), padding=1),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Dropout(0.2)
        )
        self.IMAGE_SHAPE = IMAGE_SHAPE

        with torch.no_grad():
            x = self.cnn(torch.full(self.IMAGE_SHAPE, 0, dtype=torch.float32)[None])
            n_batches, _, _, h = x.size()
            n_linear = x.view(n_batches, -1, h).shape[1]

        self.linear = nn.Linear(in_features=n_linear, out_features=64)

        self.rnn = nn.Sequential(
                # Bidirectional(64, 128, 256, lstm=True),
                Bidirectional(64, 128, num_outputs + 1, lstm=True)
        )
        
    def forward(self, inputs):
        x = self.cnn(inputs)

        n_batches, n_channels, w, h = x.size()

        x = x.view(n_batches, -1, h)
        x = x.permute(0, 2, 1)

        x = self.linear(x)

        x = x.permute(1, 0, 2)
        x = self.rnn(x)

        # x = self.lstm_2(x)
        # x = self.fc2(x)
        return x

def save_checkpoint( model, IMAGE_SHAPE):
    path = "./checkpoint"
    isdir = os.path.isdir(path)
    if isdir:
        PATH = path+"/model.pt"
    else:
        os.mkdir(path)
        PATH = path+"/model.pt"
        
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "IMAGE_SHAPE": IMAGE_SHAPE
    }
    torch.save(checkpoint, PATH)

def load_checkpoint(PATH):
    checkpoint = torch.load(PATH)
    return checkpoint