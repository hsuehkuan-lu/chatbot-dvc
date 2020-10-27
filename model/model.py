from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ChatbotEncoder(BaseModel):
    def __init__(self, vocab_size, padding_idx, hidden_size, embed_size, n_layers=1, dropout=0.):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=n_layers,
                          bidirectional=True, dropout=dropout)

    def forward(self, input_seq, input_lengths, hidden=None):
        """
        output = [seq_len, batch_size, num_directions * hidden_size]
        hidden = [num_layers * num_directions, batch_size, hidden_size]
        """
        emb = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(emb, input_lengths, enforce_sorted=False)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # out = nn.Linear(2 * self.hidden_size, self.hidden_size)
        # outputs = F.relu(out(outputs))
        return outputs, hidden


class Attention(BaseModel):
    def __init__(self, method, hidden_size):
        super().__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(2 * self.hidden_size, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=-1)

    def general_score(self, hidden, encoder_output):
        attn = self.attn(encoder_output)
        return torch.sum(hidden * attn, dim=-1)

    def concat_score(self, hidden, encoder_output):
        attn = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1),
                                    encoder_output), dim=-1)).tanh()
        return torch.sum(self.v * attn, dim=-1)

    def forward(self, hidden, encoder_outputs):
        attn_energies = None
        if self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()
        # [batch_size, max_length]
        # return: [batch_size, 1, max_length]
        return F.softmax(attn_energies, dim=-1).unsqueeze(1)


class LuongAttnDecoderRNN(BaseModel):
    def __init__(self, attn_model, embedding, embed_size, hidden_size, vocab_size, n_layers=1, dropout=0.1):
        super().__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.embed_size = embed_size
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(2 * hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

        self.attn = Attention(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        """
        input_step: [1, batch_size] (word token (not converted))
        last_hidden: [n_layers * num_directions, batch_size, hidden_size]
        encoder_outputs: [max_length, batch_size, hidden_size]

        return: output [batch_size, voc.num_words]
                hidden [n_layers * num_directions, batch_size, hidden_size]
        """
        emb = self.embedding(input_step)
        emb = self.embedding_dropout(emb)
        # output = []
        output, hidden = self.gru(emb, last_hidden)
        attn_weights = self.attn(output, encoder_outputs)
        # [batch_size, 1, max_length] * [batch_size, max_length, hidden_size]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # [seq_len, batch_size, 2 * hidden_size]
        # output = [1, batch_size, hidden_size] -> [batch_size, hidden_size]
        output = output.squeeze(0)
        # [batch_size, hidden_size]
        context = context.squeeze(1)
        concat_input = torch.cat((output, context), dim=-1)
        concat_output = torch.tanh(self.concat(concat_input))

        output = self.out(concat_output)
        output = F.softmax(output, dim=-1)
        return output, hidden


class GreedySearchDecoder(BaseModel):
    def __init__(self, encoder, decoder, init_idx):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.init_idx = init_idx

    def forward(self, talk_seq, talk_seq_len, sent_len):
        encoder_outputs, encoder_hidden = self.encoder(talk_seq, talk_seq_len)
        decoder_hidden = encoder_hidden[-self.decoder.n_layers:]
        decoder_input = torch.ones(1, 1, dtype=torch.long) * self.init_idx

        all_tokens = []
        all_scores = []
        # all_tokens = torch.zeros([0], dtype=torch.long)
        # all_scores = torch.zeros([0])

        for _ in range(sent_len):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_scores, decoder_input = torch.max(decoder_output, dim=-1)
            all_tokens += [decoder_input]
            all_scores += [decoder_scores]
            # all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            # all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, dim=0)

        all_tokens = torch.stack(all_tokens, dim=0)
        all_scores = torch.stack(all_scores, dim=0)
        return all_tokens, all_scores
