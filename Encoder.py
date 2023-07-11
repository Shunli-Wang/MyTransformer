import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

def subseqent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape, k=1)).astype('uint8')
    return torch.from_numpy(1 - subsequent_mask) # 下三角矩阵，包含主对角线

    # array([[1, 0, 0, 0, 0],
    #        [1, 1, 0, 0, 0],
    #        [1, 1, 1, 0, 0],
    #        [1, 1, 1, 1, 0],
    #        [1, 1, 1, 1, 1]])

    # 横坐标：目标词汇位置
    # 纵坐标：可以查看的位置，看纵轴的0位置，目前无可见词汇；纵轴的1位置，目前可以看到0位置的词汇
    # 掩码作用：特定情况下不能使用未来信息。

def attention(query, key, value, mask=None, dropout=None):

    # SingleHeadSetting: query (B, L, C)
    # MultiHeadSetting: query (B, Head, L, d_k)
    d_k = query.size(-1)

    # SingleHeadSetting: scores (B, L, L)
    # MultiHeadSetting: scores (B, Head, L, L)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # SingleHeadSetting: mask (L, L)
        # MultiHeadSetting: mask (1, Head, L, L)
        scores = scores.masked_fill(mask == 0, -1e9) # -Infinity to 0
    # array([[1, -, -, -, -],
    #        [1, 1, -, -, -],
    #        [1, 1, 1, -, -],
    #        [1, 1, 1, 1, -],
    #        [1, 1, 1, 1, 1]])

    p_atta = torch.softmax(scores, dim=-1)

    if dropout is not None:
        p_atta = dropout(p_atta)

    return torch.matmul(p_atta, value), p_atta

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % head == 0
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (B, L, C)

        if mask is not None:
            mask = mask.unsqueeze(0) # (Head, L, L) -> (1, Head, L, L)
        batch_size = query.size(0)

        query, key, value = [
            model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) 
            for model, x in zip(self.linears, (query, key, value))
            ] # (B, Head, L, C//Head)
        # Put len Dim close with the d_k Dim to learn the relation.

        # attn: (B, Head, L, C//Head) * (B, Head, C//Head, L) => (B, Head, L, L)
        # x: (B, Head, L, C//Head)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # x: (B, Head, L, C//Head) -> (B, L, C)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        # contiguous() after transpose, before view

        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # d_model -> d_ff -> d_model
        return self.w2(self.dropout(F.relu(self.w1(x))))

class LayerNorm(nn.Module):
    def __init__(self, featDim, eps=1e-6):
        # featDim: dimNum of the last dim.
        # eps: prevent /0
        super(LayerNorm, self).__init__()

        self.a2 = nn.Parameter(torch.ones(featDim)) # (C,)
        self.b2 = nn.Parameter(torch.zeros(featDim)) # (C,)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True) # (B, L, C) -> (B, L, 1)
        std = x.std(-1, keepdim=True) # (B, L, C) -> (B, L, 1)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
    
    def forward(self, x, sublayer):
        # sublayer = lambda x: self_attn(x, x, x, mask) when the self_attn is passed in.
        return x + self.dropout(self.norm(sublayer(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.d_model = d_model

        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x:self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

if __name__ == '__main__':
    print('-' * 50)
    B, L, C, numHeads = 2, 4, 512, 8

    # LayerNorm Testing
    x = torch.randn(B, L, C)
    layerNorm = LayerNorm(featDim=C, eps=1e-6)
    print('Before LayerNorm: \t x.shape: ', x.shape)
    print('After LayerNorm: \t x.shape: ', layerNorm(x).shape)
    print('-' * 50)

    # attention func Testing
    x = torch.randn(B, L, C)
    print('Before attention func: \t x.shape: ', x.shape)
    print('After attention func: \t x.shape: ', attention(x, x, x)[0].shape)
    print('-' * 50)

    # PositionwiseFeedForward Testing
    x = torch.randn(B, L, C)
    positionwiseFeedForward = PositionwiseFeedForward(d_model=C, d_ff=C//4, dropout=0.1)
    print('Before PositionwiseFeedForward func: \t x.shape: ', x.shape)
    print('After PositionwiseFeedForward func: \t x.shape: ', positionwiseFeedForward(x).shape)
    print('-' * 50)

    # MultiHeadedAttention Testing
    x = torch.randn(B, L, C)
    selfAttnMultiHead = MultiHeadedAttention(head=numHeads, embedding_dim=C, dropout=0.1)
    mask = torch.randn(numHeads, L, L)
    print('Before MultiHeadedAttention: \t x.shape: ', x.shape)
    print('After MultiHeadedAttention: \t x.shape: ', selfAttnMultiHead(x, x, x, mask).shape)
    print('-' * 50)

    # SublayerConnection Testing
    x = torch.randn(B, L, C)
    subLayerConnection = SublayerConnection(d_model=C, dropout=0.1)
    selfAttnMultiHead = MultiHeadedAttention(head=numHeads, embedding_dim=C, dropout=0.1)
    mask = torch.randn(numHeads, L, L)
    sublayer = lambda x: selfAttnMultiHead(x, x, x, mask) # Adapting the multiHeadAttLayer
    print('Before SublayerConnection: \t x.shape: ', x.shape)
    print('After SublayerConnection: \t x.shape: ', subLayerConnection(x, sublayer).shape)
    print('-' * 50)

    # EncoderLayer Testing
    x = torch.randn(B, L, C)
    self_attn = MultiHeadedAttention(numHeads, embedding_dim=C, dropout=0.1)
    ff = PositionwiseFeedForward(d_model=C, d_ff=C//4, dropout=0.1)
    c = copy.deepcopy
    mask = torch.zeros(numHeads, L, L)
    encoderLayer = EncoderLayer(d_model=C, self_attn=self_attn, feed_forward=ff, dropout=0.1) # DeepCopy ?
    print('Before EncoderLayer: \t x.shape: ', x.shape)
    print('After EncoderLayer: \t x.shape: ', encoderLayer(x, mask).shape)
    print('-' * 50)

    # Encoder Testing
    x = torch.randn(B, L, C)
    self_attn = MultiHeadedAttention(numHeads, embedding_dim=C, dropout=0.1)
    ff = PositionwiseFeedForward(d_model=C, d_ff=C//4, dropout=0.1)
    mask = torch.zeros(numHeads, L, L)
    encoderLayer = EncoderLayer(d_model=C, self_attn=self_attn, feed_forward=ff, dropout=0.1) # DeepCopy ?
    encoder = Encoder(layer=encoderLayer, N=8)
    print('Before Encoder: \t x.shape: ', x.shape)
    print('After Encoder: \t\t x.shape: ', encoder(x, mask).shape)
    print('-' * 50)

# ###############################
# QKV的形象化理解：
# query: 给定全文内容，篇幅很长，待提取文本
# key: 老师给定的参考关键词，具有一定的指向信息，但是并不是最精准的信息
# value: 自己通过阅读得到的脑海印象，起初是和key一致的，但是后期会形成自己的答案
# 一般而言key和value默认是一致的，若Q=K=V则是自注意力机制，即对自身文本进行一次特征提取

# 多head目的：让每个注意力机制去优化每个词汇的不同特征部分，均衡同一种注意力机制可能产生的偏差，词义有更多元的表达
