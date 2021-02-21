import torch
import torch.nn as nn
from torchtext.vocab import GloVe
from data.dataset_helper import TEXT, get_data_iter
from data.label_parser import label_cnt
import random
from model.params import DEVICE, DEVICE_ORDER

class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        vocab = TEXT.vocab
        self.embed = nn.Embedding(len(vocab), embedding_size)
        self.embed.weight.data.copy_(vocab.vectors)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p, bidirectional=True)
        self.hidden_fc = nn.Linear(hidden_size * 2, hidden_size) # reduce hidden dimension
        self.cell_fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # index -> (seq_len, batch_size)
        embedding = self.embed(x)
        # embedding -> (seq_len, batch, embedding_size)
        encoder_states, (hidden, cell) = self.lstm(embedding)
        # (num_layers * num_dir, batch, hidden_size) -> num_dir * (num_layers, batch, hidden_size) -> (num_layers, batch, hidden_size * num_dir)
        hidden_forward = torch.cat([hidden[i * 2: i * 2 + 1] for i in range(self.num_layers)], dim=0)
        hidden_backward = torch.cat([hidden[i * 2 + 1: i * 2 + 2] for i in range(self.num_layers)], dim=0)
        hidden = torch.cat([hidden_forward, hidden_backward], dim=2)

        cell_forward = torch.cat([cell[i * 2: i * 2 + 1] for i in range(self.num_layers)], dim=0)
        cell_backward = torch.cat([cell[i * 2 + 1: i * 2 + 2] for i in range(self.num_layers)], dim=0)
        cell = torch.cat([cell_forward, cell_backward], dim=2)

        hidden = self.hidden_fc(hidden)
        cell = self.cell_fc(cell)
        # encoder_state -> (seq_len, batch, hidden_size * num_dir)
        # cell, hidden -> (num_layers, batch, hidden_size)
        return encoder_states, hidden, cell

class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, p):
        super(Decoder, self).__init__()
        self.energy = nn.Linear(hidden_size * 3, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.lstm = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers, dropout=p)

        self.scr_classifier = nn.Linear(hidden_size, 6)
        self.cpu_classifier = nn.Linear(hidden_size, 10)
        self.ram_classifier = nn.Linear(hidden_size, 6)
        self.hdk_classifier = nn.Linear(hidden_size, 10)
        self.gcd_classifier = nn.Linear(hidden_size, 8)
        self.classifier = {
            "screen": self.scr_classifier,
            "cpu": self.cpu_classifier,
            "ram": self.ram_classifier,
            "hdisk": self.hdk_classifier,
            "gcard": self.gcd_classifier
        }
        self.ini_embedding = nn.Embedding(1, embedding_size)
        self.scr_embedding = nn.Embedding(6, embedding_size)
        self.cpu_embedding = nn.Embedding(10, embedding_size)
        self.ram_embedding = nn.Embedding(6, embedding_size)
        self.hdk_embedding = nn.Embedding(10, embedding_size)
        self.gcd_embedding = nn.Embedding(8, embedding_size)
        self.task_embedding = {
            "init": self.ini_embedding,
            "screen": self.scr_embedding,
            "cpu": self.cpu_embedding,
            "ram": self.ram_embedding,
            "hdisk": self.hdk_embedding,
            "gcard": self.gcd_embedding
        }

    # hidden and cell are from previous time step of decoder for t > 1.
    # for t = 1, they are from the last step of encoder
    def forward(self, x, encoder_states, hidden, cell, input_task, cur_task):
        seq_len = encoder_states.shape[0]
        # (num_layers, batch, hidden_size) -> (1, batch, hidden_size) -> (seq_len * num_layers, batch, hidden_size)
        h_reshaped = hidden[-1].repeat(seq_len, 1, 1)

        # (seq_len, batch, hidden_size + hidden_size * 2) -> (seq_len, batch, 1)
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        attention_score = self.softmax(energy)
        # (seq_len, batch, 1) * (seq_len, batch, hidden_size * num_dir) -> (1, batch, hidden_size * num_dir)
        context_vector = torch.einsum("snk,snl->knl", attention_score, encoder_states)

        # add a dimension as seq_len
        x = x.unsqueeze(0)
        # index -> (1, batch, embedding_size)
        embedding = self.task_embedding[input_task](x)
        # (1, batch, hidden_size * 2 + task_embedding_size)
        input_lstm = torch.cat((context_vector, embedding), dim=2)

        # output -> (1, batch, hidden_size)
        output, (hidden, cell) = self.lstm(input_lstm, (hidden, cell))
        # prediction -> (batch, task_class_num)
        predictions = self.classifier[cur_task](output).squeeze(0)
        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, p):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_force = p

    def forward(self, source, target_dict):
        batch_size = source.shape[1]
        encoder_states, hidden, cell = self.encoder(source)
        outputs = []
        prev_task = "init"
        x = torch.zeros(batch_size, device=DEVICE, dtype=torch.int64)
        for t, task in enumerate(DEVICE_ORDER):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell, prev_task, task)
            x = output.argmax(1) if random.random() < self.teacher_force else target_dict[task]
            outputs.append(output)
            prev_task = task
        return outputs

#
# def check_embedding():
#     _, _, _, _, need_iter = get_data_iter()
#     embedding_glove = GloVe(name='6B', dim=100)
#     vocab = TEXT.vocab
#     embed = nn.Embedding(len(vocab), 100)
#     embed.weight.data.copy_(vocab.vectors)
#     for batch_index, batch in enumerate(need_iter):
#         glove_s = []
#         for j in range(batch.text.shape[1]):
#             text = [TEXT.vocab.itos[batch.text.numpy()[i][j]] for i in range(batch.text.shape[0])]
#             print(text)
#             s_e = 0.0
#             for token in text:
#                 s_e += sum(embedding_glove[token])
#             glove_s.append(s_e)
#         embedding = embed(batch.text)
#         print(embedding.shape)
#         e = torch.einsum("ijk -> j", embedding)
#         print(e)
#         print(glove_s)
#         break
#
# def check_encoder():
#     _, _, _, _, data_iter = get_data_iter()
#     embedding_size = 100
#     hidden_size = 512
#     num_layers = 1
#     encoder = Encoder(embedding_size, hidden_size, num_layers, p=0.5)
#     for _, batch in enumerate(data_iter):
#         data = batch.text
#         (seq_len, batch_size) = data.shape
#         encoder_state, hidden, cell = encoder(data)
#         print(encoder_state.shape, hidden.shape, cell.shape)
#         assert encoder_state.shape == (seq_len, batch_size, hidden_size * 2)
#         assert hidden.shape == (1, batch_size, hidden_size)
#         assert cell.shape == (1, batch_size, hidden_size)
#         break
#     print("encoder test pass")
#
# def check_decoder():
#     _, _, _, _, data_iter = get_data_iter()
#     embedding_size = 100
#     hidden_size = 512
#     num_layers = 1
#     encoder = Encoder(embedding_size, hidden_size, num_layers, p=0.5)
#     decoder = Decoder(embedding_size, hidden_size, num_layers, 0.5)
#     task = "cpu"
#     for _, batch in enumerate(data_iter):
#         data = batch.text
#         (seq_len, batch_size) = data.shape
#         encoder_state, hidden, cell = encoder(data)
#         x = torch.zeros((batch_size, 1))
#         predictions, hidden, cell = decoder(x, encoder_state, cell, hidden, "init", task)
#         assert predictions.shape == (batch_size, label_cnt[task])
#         assert hidden.shape == (1, batch_size, hidden_size)
#         assert cell.shape == (1, batch_size, hidden_size)
#         break
#     print("decoder test pass")
#
# def check_seq2seq():
#     _, _, _, _, data_iter = get_data_iter()
#     embedding_size = 100
#     hidden_size = 512
#     num_layers = 1
#     encoder = Encoder(embedding_size, hidden_size, num_layers, p=0.5)
#     decoder = Decoder(embedding_size, hidden_size, num_layers, 0.5)
#     model = Seq2Seq(encoder, decoder)
#     for _, batch in enumerate(data_iter):
#         samples = batch.text
#         (seq_len, batch_size) = samples.shape
#         outputs = model(samples)
#         for index, device in enumerate(devices):
#             assert outputs[index].shape == (batch_size, label_cnt[device])
#         break
#     print("test seq2seq pass")
