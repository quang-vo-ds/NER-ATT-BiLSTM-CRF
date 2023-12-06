import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.layers.layer_base import LayerBase

class LayerAttention(LayerBase):
    def __init__(self, gpu, hidden_dim):
        super(LayerAttention, self).__init__(gpu)
        self.hidden_dim = hidden_dim
        self.att_weights = nn.Parameter(torch.Tensor(1, self.hidden_dim))
        self.output_dim = hidden_dim
        stdv = 1.0 / np.sqrt(self.hidden_dim)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def is_cuda(self):
        return self.att_weights.is_cuda

    def forward(self, input_tensor, mask_tensor):
        batch_size, max_len = input_tensor.size()[:2]
        # apply attention layer
        weights = torch.bmm(input_tensor,
                            self.att_weights  # (1, hidden_dim)
                            .permute(1, 0)  # (hidden_dim, 1)
                            .unsqueeze(0)  # (1, hidden_dim, 1)
                            .repeat(batch_size, 1, 1) # (batch_size, hidden_dim, 1)
                            ) # (batch_size, max_seq_len, 1)
        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)
        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask_tensor
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row
        attentions = masked.div(_sums)
        # apply attention weights
        weighted = torch.mul(input_tensor, attentions.unsqueeze(-1).expand_as(input_tensor))
        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()
        return weighted