import torch
import torch.nn as nn
from src.layers.layer_birnn_base import LayerBiRNNBase

class LayerBiLSTM(LayerBiRNNBase):
    """BiLSTM layer implements standard bidirectional LSTM recurrent layer"""
    def __init__(self, input_dim, hidden_dim, gpu):
        super(LayerBiLSTM, self).__init__(input_dim, hidden_dim, gpu)
        self.num_layers = 1
        self.num_directions = 2
        rnn = nn.LSTM(input_size=input_dim,
                      hidden_size=hidden_dim,
                      num_layers=1,
                      batch_first=True,
                      bidirectional=True)
        self.rnn = rnn

    def lstm_custom_init(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse)
        self.rnn.bias_hh_l0.data.fill_(0)
        self.rnn.bias_hh_l0_reverse.data.fill_(0)
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_ih_l0_reverse.data.fill_(0)
        # Init forget gates to 1
        for names in self.rnn._all_weights:
            for name in filter(lambda n: 'bias' in n, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def forward(self, input_tensor, mask_tensor): #input_tensor shape: batch_size x max_seq_len x dim
        batch_size, max_seq_len, _ = input_tensor.shape
        input_packed, reverse_sort_index = self.pack(input_tensor, mask_tensor)
        h0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        c0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        output_packed, _ = self.rnn(input_packed, (h0, c0))
        output_tensor = self.unpack(output_packed, max_seq_len, reverse_sort_index)
        return output_tensor  # shape: batch_size x max_seq_len x hidden_dim*2

    def is_cuda(self):
        return self.rnn.weight_hh_l0.is_cuda