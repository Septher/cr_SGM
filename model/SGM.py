import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p, bidirectional=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell



class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, p):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)


    def forward(self):
        return
