import torch
import torch.nn as nn
from torchtext.vocab import GloVe
from data.dataset_helper import TEXT, devices, get_data_iter
from data.label_parser import label_cnt

class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        vocab = TEXT.vocab
        self.embed = nn.Embedding(len(vocab), embedding_size)
        self.embed.weight.data.copy_(vocab.vectors)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p, bidirectional=True)
        self.hidden_fc = nn.Linear(hidden_size * num_layers * 2, hidden_size) # reduce hidden dimension
        self.cell_fc = nn.Linear(hidden_size * num_layers * 2, hidden_size)

    def forward(self, x):
        # x -> (seq_len, batch_size)
        embedding = self.embed(x)
        # embedding -> (seq_len, batch_size, embedding_size)
        encoder_states, (hidden, cell) = self.lstm(embedding)
        hidden = self.hidden_fc(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.cell_fc(torch.cat((cell[0:1], cell[1:2]), dim=2))
        # encoder_state -> (seq_len, batch_size, hidden_size * num_layers * 2)
        # cell, hidden -> (batch_size, hidden_size)
        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, p):
        super(Decoder, self).__init__()
        self.energy = nn.Linear(hidden_size * 3, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.lstm = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers, dropout=p)
        self.classifier = {
            "screen": nn.Linear(hidden_size, 6),
            "cpu": nn.Linear(hidden_size, 10),
            "ram": nn.Linear(hidden_size, 6),
            "hdisk": nn.Linear(hidden_size, 10),
            "gcard": nn.Linear(hidden_size, 8)
        }
        self.task_embedding = {
            "init": nn.Linear(1, embedding_size),
            "screen": nn.Linear(6, embedding_size),
            "cpu": nn.Linear(10, embedding_size),
            "ram": nn.Linear(6, embedding_size),
            "hdisk": nn.Linear(10, embedding_size),
            "gcard": nn.Linear(8, embedding_size)
        }

    # hidden and cell are from previous time step of decoder for t > 1.
    # for t = 1, they are from the last step of encoder
    def forward(self, x, encoder_states, hidden, cell, input_task, cur_task):
        # add a dimension for seq_len, x -> (1, batch_size)
        x = x.unsqueeze(0)
        embedding = self.task_embedding[input_task](x)
        # (1, batch_size, embedding_size)

        seq_len = encoder_states.shape[0]
        h_reshaped = hidden.repeat(seq_len, 1, 1)
        # (seq_len, batch_size, hidden_size)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # (seq_len, batch_size, 1)
        attention = self.softmax(energy)
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)
        # (1, batch_size, hidden_size * 2)
        input_lstm = torch.cat((context_vector, embedding), dim=2)
        # (1, batch_size, hidden_size * 2 + task_embedding_size)

        output, (hidden, cell) = self.lstm(input_lstm, (hidden, cell))
        # output -> (1, batch_size, hidden_size)
        predictions = self.classifier[cur_task](output).squeeze(0)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, samples, task_dict):
        batch_size = samples.shape[1]
        encoder_states, hidden, cell = self.encoder(samples)
        outputs = []
        prev_task = "init"
        x = torch.zeros((batch_size, 1))
        for t, task in enumerate(devices):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell, prev_task, task)
            x = output.argmax(1)
            x = self.to_one_hot(x, task)
            outputs.append(output)
            prev_task = task
        return outputs

    def to_one_hot(self, x, task):
        batch_size, dimension = x.shape[0], label_cnt[task]
        one_hot = torch.zeros((batch_size, dimension))
        x = x.unsqueeze(1)
        # (batch_size, 1)
        return one_hot.scatter_(1, x, 1)


def check_embedding():
    _, _, _, _, need_iter = get_data_iter()
    embedding_glove = GloVe(name='6B', dim=100)
    vocab = TEXT.vocab
    embed = nn.Embedding(len(vocab), 100)
    embed.weight.data.copy_(vocab.vectors)
    for batch_index, batch in enumerate(need_iter):
        glove_s = []
        for j in range(batch.text.shape[1]):
            text = [TEXT.vocab.itos[batch.text.numpy()[i][j]] for i in range(batch.text.shape[0])]
            print(text)
            s_e = 0.0
            for token in text:
                s_e += sum(embedding_glove[token])
            glove_s.append(s_e)
        embedding = embed(batch.text)
        print(embedding.shape)
        e = torch.einsum("ijk -> j", embedding)
        print(e)
        print(glove_s)
        break

def check_encoder():
    _, _, _, _, data_iter = get_data_iter()
    embedding_size = 100
    hidden_size = 512
    num_layers = 1
    encoder = Encoder(embedding_size, hidden_size, num_layers, p=0.5)
    for _, batch in enumerate(data_iter):
        data = batch.text
        (seq_len, batch_size) = data.shape
        encoder_state, hidden, cell = encoder(data)
        print(encoder_state.shape, hidden.shape, cell.shape)
        assert encoder_state.shape == (seq_len, batch_size, hidden_size * 2)
        assert hidden.shape == (1, batch_size, hidden_size)
        assert cell.shape == (1, batch_size, hidden_size)
        break
    print("encoder test pass")

def check_decoder():
    _, _, _, _, data_iter = get_data_iter()
    embedding_size = 100
    hidden_size = 512
    num_layers = 1
    encoder = Encoder(embedding_size, hidden_size, num_layers, p=0.5)
    decoder = Decoder(embedding_size, hidden_size, num_layers, 0.5)
    task = "cpu"
    for _, batch in enumerate(data_iter):
        data = batch.text
        (seq_len, batch_size) = data.shape
        encoder_state, hidden, cell = encoder(data)
        x = torch.zeros((batch_size, 1))
        predictions, hidden, cell = decoder(x, encoder_state, cell, hidden, "init", task)
        assert predictions.shape == (batch_size, label_cnt[task])
        assert hidden.shape == (1, batch_size, hidden_size)
        assert cell.shape == (1, batch_size, hidden_size)
        break
    print("decoder test pass")

def check_seq2seq():
    _, _, _, _, data_iter = get_data_iter()
    embedding_size = 100
    hidden_size = 512
    num_layers = 1
    encoder = Encoder(embedding_size, hidden_size, num_layers, p=0.5)
    decoder = Decoder(embedding_size, hidden_size, num_layers, 0.5)
    model = Seq2Seq(encoder, decoder)
    for _, batch in enumerate(data_iter):
        samples = batch.text
        (seq_len, batch_size) = samples.shape
        task_dict = {
            "cpu": batch.cpu,
            "ram": batch.ram,
            "screen": batch.screen,
            "hdisk": batch.hdisk,
            "gcard": batch.gcard
        }
        # (seq_len, batch_size) = data.shape
        outputs = model(samples, task_dict)
        for index, device in enumerate(devices):
            assert outputs[index].shape == (batch_size, label_cnt[device])
        break
    print("test seq2seq pass")

check_seq2seq()
