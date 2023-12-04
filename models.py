import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TCoN(nn.Module):
    def __init__(self, num_f=0):
        super().__init__()
        self.emb = nn.Linear(in_features=num_f, out_features=200)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(200,100)
        self.attention = nn.MultiheadAttention(100, 2, batch_first=True, dropout=0.15)
        self.gru = nn.GRU(input_size=100,hidden_size=2,num_layers=1,batch_first=True)

    def forward(self, input_tuple):
        seqs, lengths = input_tuple
        x = self.relu(self.emb(seqs))
        x = self.linear(x)
        x, weights = self.attention(x, x, x)
        x = pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.gru(x)
        seqs, lengths = pad_packed_sequence(x, batch_first=True)

        prob = []
        c = 0
        for i in lengths:
            prob.append(seqs[c,i-1,:])
            c +=1
        result = torch.stack(prob,dim=0)
        return result
