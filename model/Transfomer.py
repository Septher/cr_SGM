import torch.nn as nn
import math
from model.params import *
from data.dataset_helper import TEXT

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=EMBEDDING_SIZE
        )
        vocab = TEXT.vocab
        self.word_embed = nn.Embedding(len(vocab), EMBEDDING_SIZE)
        self.word_embed.weight.data.copy_(vocab.vectors)
        self.pad_idx = TEXT.vocab.stoi["<pad>"]
        self.ini_embedding = nn.Embedding(1, EMBEDDING_SIZE)
        self.scr_embedding = nn.Embedding(6, EMBEDDING_SIZE)
        self.cpu_embedding = nn.Embedding(10, EMBEDDING_SIZE)
        self.ram_embedding = nn.Embedding(6, EMBEDDING_SIZE)
        self.hdk_embedding = nn.Embedding(10, EMBEDDING_SIZE)
        self.gcd_embedding = nn.Embedding(8, EMBEDDING_SIZE)
        self.task_embedding = {
            "init": self.ini_embedding,
            "screen": self.scr_embedding,
            "cpu": self.cpu_embedding,
            "ram": self.ram_embedding,
            "hdisk": self.hdk_embedding,
            "gcard": self.gcd_embedding
        }

        self.scr_classifier = nn.Linear(EMBEDDING_SIZE, 6)
        self.cpu_classifier = nn.Linear(EMBEDDING_SIZE, 10)
        self.ram_classifier = nn.Linear(EMBEDDING_SIZE, 6)
        self.hdk_classifier = nn.Linear(EMBEDDING_SIZE, 10)
        self.gcd_classifier = nn.Linear(EMBEDDING_SIZE, 8)
        self.classifier = {
            "screen": self.scr_classifier,
            "cpu": self.cpu_classifier,
            "ram": self.ram_classifier,
            "hdisk": self.hdk_classifier,
            "gcard": self.gcd_classifier
        }

        self.pos_encoder = PositionalEncoding(d_model=EMBEDDING_SIZE)

    def calculate_trg_embedding(self, target_dict, batch):
        # shifted right
        x = torch.zeros((1, batch), device=DEVICE, dtype=torch.int64)
        bos = self.task_embedding["init"](x)
        embeddings = [bos]
        # target sequence
        for _, task in enumerate(DEVICE_ORDER):
            original_input = target_dict[task].unsqueeze(0)
            embedding = self.task_embedding[task](original_input)
            embeddings.append(embedding)
        return torch.cat(embeddings, dim=0)

    # bidirectional mask
    def get_trg_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
                Unmasked positions are filled with float(0.0).
            """
        mask = torch.ones(sz, sz)
        for i in range(sz - 1):
            mask[i][i + 1] = 0
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def classify_output(self, outputs):
        result = []
        for t, task in enumerate(DEVICE_ORDER):
            output = self.classifier[task](outputs[t])
            result.append(output)
        return result

    def forward(self, src, target_dict):
        _, batch = src.shape
        # (batch, seq_len)
        src_padding_mask = (src.transpose(0, 1) == self.pad_idx).to(DEVICE)
        # (1 + output_len, 1 + output_len)
        trg_mask = self.transformer.generate_square_subsequent_mask(1 + len(target_dict)).to(DEVICE)
#        trg_mask = self.get_trg_mask(1 + len(target_dict))
        # (input_seq_len, batch, word_embedding_size)
        src_input = self.pos_encoder(self.word_embed(src))
        # (output_seq_len, batch, word_embedding_size)
        trg_input = self.calculate_trg_embedding(target_dict, batch)
        # (output_seq_len, batch, word_embedding_size)
        outputs = self.transformer(src=src_input, tgt=trg_input, tgt_mask=trg_mask, src_key_padding_mask=src_padding_mask)
        # remove the last output which is the <eos>
        outputs = outputs[:-1]
        return self.classify_output(outputs)