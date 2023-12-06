import torch
import torch.nn as nn
from src.layers.layer_base import LayerBase

class LayerCharBiLSTM(LayerBase):
    """LayerCharCNN implements character-level convolutional 1D layer."""
    def __init__(self, gpu, char_embeddings_dim, char_hidden_dim):
        super(LayerCharBiLSTM, self).__init__(gpu)
        self.char_embeddings_dim = char_embeddings_dim
        self.char_hidden_dim = char_hidden_dim
        self.output_dim = 2 * char_hidden_dim
        self.lstm = nn.LSTM(input_size=char_embeddings_dim,
                            hidden_size=char_hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

    def is_cuda(self):
        return self.lstm.weight_hh_l0.is_cuda

    def forward(self, char_embeddings_feature): # batch_num x max_seq_len x char_embeddings_dim x word_len
        batch_num, max_seq_len, char_embeddings_dim, word_len = char_embeddings_feature.shape
        output_tensor = self.tensor_ensure_gpu(torch.zeros(batch_num, max_seq_len, self.output_dim, dtype=torch.float))
        for k in range(max_seq_len):
            input_packed = char_embeddings_feature[:,k,:,:].permute(0,2,1)
            output_pack, _ =  self.lstm(input_packed)
            output_tensor[:,k,:] = output_pack[:,-1,:]
        return output_tensor  # shape: batch_size x max_seq_len x hidden_dim*2