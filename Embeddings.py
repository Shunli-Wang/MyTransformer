import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)
        # Initialize position matrix according to the max_len & d_model
        pe = torch.zeros(max_len, d_model, requires_grad=False) # (maxLen, 512)
        # Absolute position matrix
        position = torch.arange(0, max_len).unsqueeze(1) # (maxLen, 1) [[0], [1], ..., [maxLen-1]]
        # Transform matrix, stride as 2, frequency of signals.
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        ) # (256,)

        # sin & cos in different frequencies 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, 5000, 512) 

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)

if __name__ == '__main__':
    print('-' * 50)

    # Embeddings Testing
    d_model, vocab = 512, 1000
    x = torch.tensor([[100, 2, 421, 508], [491, 998, 1, 221]]) # word2index: (2, 4) 
    embLayer = Embeddings(d_model, vocab) # [2, 4, 512]
    print('Before Embeddings: \t x.shape: ', x.shape)
    print('After Embeddings: \t x.shape: ', embLayer(x).shape)
    print('-' * 50)

    # PositionalEncoding Testing
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(d_model=20, dropout=0) # 模型维度为20
    y = pe(torch.zeros(1, 100, 20)) # 传入0，只展示pe

    plt.plot(np.arange(100), y[0, :, 7:13].numpy())
    plt.legend(["dim %d"%p for p in range(7, 13)])
    plt.show()
