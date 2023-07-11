import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from Embeddings import *
from Encoder import *
from Decoder import *

from pyitcast.transformer_utils import Batch, get_std_opt, LabelSmoothing, SimpleLossCompute, run_epoch, greedy_decode, run_epoch

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)
    
    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)
    
    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)

def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.8):
    c = copy.deepcopy

    attn = MultiHeadedAttention(head, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N), 
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)), 
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)), 
        Generator(d_model, target_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model

def data_generator(V, batch_size, num_batch):
    # V: max_num + 1
    for i in range(num_batch):
        data = torch.LongTensor(np.random.randint(1, V, size=(batch_size, 10)))
        data[:, 0] = 1 # Start sign 

        source = data.requires_grad_(False)
        target = data.requires_grad_(False)

        yield Batch(source, target)

def run(model, loss, epochs=10):
    for epoch in range(epochs):
        # Train
        model.train()
        run_epoch(data_generator(11, 8, 20), model, loss)

        # Eval
        model.eval()
        run_epoch(data_generator(11, 8, 5), model, loss)

    model.eval()
    source = torch.LongTensor([[1,3,2,5,4,6,7,8,9,10]])
    source_mask = torch.ones(1,1,10)

    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)

if __name__ == '__main__':
    source_vocab = 11
    target_vocab = 11
    N = 6

    model = make_model(source_vocab, target_vocab, N)
    print(model)

    model_optimizer = get_std_opt(model)
    criterion = LabelSmoothing(size=target_vocab, padding_idx=0, smoothing=0.0) # (B, target_vocab) after softmax
    loss = SimpleLossCompute(model.generator, criterion, model_optimizer)

    # data_generator Testing
    V = 11
    batch_size = 20
    num_batch = 30
    res = data_generator(V, batch_size, num_batch)
    print(res)

    # run Testing
    run(model, loss, 10)
    
