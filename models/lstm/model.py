import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):

    def __init__(self, input_size, embedding_dim, hidden_dim, output_size, num_layers=1):
        super(LSTMClassifier, self).__init__()

        #self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_size = input_size

        #self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_size)

        self.dropout_layer = nn.Dropout(p=0.2)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)),
                autograd.Variable(torch.randn(1, batch_size, self.hidden_dim)))

    def forward(self, batch):
        '''self.hidden = self.init_hidden(batch.size(-1))
        batch = batch.long()
        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)'''

        outputs, (ht, ct) = self.lstm(batch)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.fc(output)

        return output