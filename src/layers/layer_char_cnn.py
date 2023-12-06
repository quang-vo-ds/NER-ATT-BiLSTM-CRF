import torch
import torch.nn as nn
from src.layers.layer_base import LayerBase

class LayerCharCNN(LayerBase):
    """LayerCharCNN implements character-level convolutional 1D layer."""
    def __init__(self, gpu, char_embeddings_dim, filter_num, char_window_size, word_len):
        super(LayerCharCNN, self).__init__(gpu)
        self.char_embeddings_dim = char_embeddings_dim
        self.char_cnn_filter_num = filter_num
        self.char_window_size = char_window_size
        self.word_len = word_len
        self.output_dim = char_embeddings_dim * filter_num
        self.conv1 = nn.Conv1d(in_channels=char_embeddings_dim,
                               out_channels=char_embeddings_dim,
                               kernel_size=char_window_size[0],
                               groups=char_embeddings_dim,
                               padding="same")
        self.conv2 = nn.Conv1d(in_channels=char_embeddings_dim,
                               out_channels=char_embeddings_dim,
                               kernel_size=char_window_size[1],
                               groups=char_embeddings_dim, 
                               padding="same")
        self.conv3 = nn.Conv1d(in_channels=char_embeddings_dim,
                               out_channels=char_embeddings_dim,
                               kernel_size=char_window_size[2],
                               groups=char_embeddings_dim, 
                               padding="same")

    def is_cuda(self):
        return self.conv1.weight.is_cuda

    def forward(self, char_embeddings_feature): # batch_num x max_seq_len x char_embeddings_dim x word_len
        batch_num, max_seq_len, char_embeddings_dim, word_len = char_embeddings_feature.shape
        max_pooling_out = self.tensor_ensure_gpu(torch.zeros(batch_num, max_seq_len, self.output_dim, dtype=torch.float))
        for k in range(max_seq_len):
            conv_out1 = self.conv1(char_embeddings_feature[:, k, :, :])
            conv_out2 = self.conv2(char_embeddings_feature[:, k, :, :])
            conv_out3 = self.conv3(char_embeddings_feature[:, k, :, :])
            conv_out = torch.cat((conv_out1, conv_out2, conv_out3), dim=1)
            max_pooling_out[:, k, :], _ = torch.max(conv_out, dim=2)
        return max_pooling_out # shape: batch_num x max_seq_len x filter_num*char_embeddings_dim