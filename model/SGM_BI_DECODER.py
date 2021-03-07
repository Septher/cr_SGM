import torch.nn as nn
import torch.nn.functional as F
from data.dataset_helper import TEXT
import random
from model.params import *

class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        vocab = TEXT.vocab
        self.embed = nn.Embedding(len(vocab), embedding_size)
        self.embed.weight.data.copy_(vocab.vectors)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p, bidirectional=True)
        self.hidden_fc = nn.Linear(hidden_size * 2, hidden_size)  # reduce hidden dimension
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

class DecoderForward(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, p, task_embedding, classifier):
        super(DecoderForward, self).__init__()
        self.energy = nn.Linear(hidden_size * 3, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.lstm = nn.LSTM(hidden_size * 2 + hidden_size * num_layers + embedding_size, hidden_size, num_layers, dropout=p)
        self.classifier = classifier
        self.task_embedding = task_embedding

    # hidden and cell are from previous time step of decoder for t > 1.
    # for t = 1, they are from the last step of encoder
    def forward(self, x, encoder_states, backward_hidden, hidden, cell, input_task, cur_task):
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
        # (1, batch, hidden_size * 2 + hidden_size + task_embedding_size)
        input_lstm = torch.cat((context_vector, backward_hidden, embedding), dim=2)

        # output -> (1, batch, hidden_size)
        output, (hidden, cell) = self.lstm(input_lstm, (hidden, cell))
        # prediction -> (batch, task_class_num)
        predictions = self.classifier[cur_task](output).squeeze(0)
        return predictions, hidden, cell


class DecoderBackward(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, p, task_embedding, classifier):
        super(DecoderBackward, self).__init__()
        self.energy = nn.Linear(hidden_size * 3, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.lstm = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers, dropout=p)

        self.classifier = classifier
        self.task_embedding = task_embedding

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
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.ini_embedding = nn.Embedding(1, TASK_EMBEDDING_SIZE)
        self.scr_embedding = nn.Embedding(6, TASK_EMBEDDING_SIZE)
        self.cpu_embedding = nn.Embedding(10, TASK_EMBEDDING_SIZE)
        self.ram_embedding = nn.Embedding(6, TASK_EMBEDDING_SIZE)
        self.hdk_embedding = nn.Embedding(10, TASK_EMBEDDING_SIZE)
        self.gcd_embedding = nn.Embedding(8, TASK_EMBEDDING_SIZE)
        self.task_embedding = {
            "init": self.ini_embedding,
            "screen": self.scr_embedding,
            "cpu": self.cpu_embedding,
            "ram": self.ram_embedding,
            "hdisk": self.hdk_embedding,
            "gcard": self.gcd_embedding
        }

        self.scr_classifier = nn.Linear(HIDDEN_SIZE, 6)
        self.cpu_classifier = nn.Linear(HIDDEN_SIZE, 10)
        self.ram_classifier = nn.Linear(HIDDEN_SIZE, 6)
        self.hdk_classifier = nn.Linear(HIDDEN_SIZE, 10)
        self.gcd_classifier = nn.Linear(HIDDEN_SIZE, 8)
        self.classifier = {
            "screen": self.scr_classifier,
            "cpu": self.cpu_classifier,
            "ram": self.ram_classifier,
            "hdisk": self.hdk_classifier,
            "gcard": self.gcd_classifier
        }

        self.teacher_force = TEACHER_FORCE

        self.encoder = Encoder(
            embedding_size=WORD_EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            p=DROP_OUT_EN
        ).to(DEVICE)

        self.decoder_forward = DecoderForward(
            embedding_size=TASK_EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            p=DROP_OUT_DE,
            task_embedding=self.task_embedding,
            classifier=self.classifier
        ).to(DEVICE)

        self.decoder_backward = DecoderBackward(
            embedding_size=TASK_EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            p=DROP_OUT_DE,
            task_embedding=self.task_embedding,
            classifier=self.classifier
        ).to(DEVICE)

    def forward(self, source, target_dict):
        batch_size = source.shape[1]
        encoder_states, hidden, cell = self.encoder(source)
        # backward
        backward_outputs = []
        prev_task = "init"
        inputs = torch.zeros(batch_size, device=DEVICE, dtype=torch.int64)
        # hidden state after the task is finished
        # (num_layers, batch, hidden_size)
        backward_hidden_dict = {}
        DEVICE_ORDER.reverse()
        for t, task in enumerate(DEVICE_ORDER):
            backward_hidden_dict[task] = torch.cat([hidden[i: i + 1] for i in range(NUM_LAYERS)], dim=2)
            output, hidden, cell = self.decoder_backward(inputs, encoder_states, hidden, cell, prev_task, task)
            inputs = output.argmax(1) if random.random() < self.teacher_force else target_dict[task]
            backward_outputs.append(output)
#            backward_hidden_dict[task] = torch.cat([hidden[i: i + 1] for i in range(NUM_LAYERS)], dim=2)
            prev_task = task
        # forward
        forward_outputs = []
        prev_task = "init"
        inputs = torch.zeros(batch_size, device=DEVICE, dtype=torch.int64)
        DEVICE_ORDER.reverse()
        for t, task in enumerate(DEVICE_ORDER):
            output, hidden, cell = self.decoder_forward(inputs, encoder_states, backward_hidden_dict[task], hidden, cell, prev_task, task)
            inputs = output.argmax(1) if random.random() < self.teacher_force else target_dict[task]
            forward_outputs.append(output)
            prev_task = task

        return forward_outputs, backward_outputs
