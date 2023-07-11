from Encoder import *

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout

        self.sublayer = clones(SublayerConnection(d_model=self.d_model, dropout=dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask)) # Q=K=V
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask)) # Q!=K=V
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)

if __name__ == '__main__':
    print('-' * 50)
    B, L, C, numHeads, dr = 2, 4, 512, 8, 0.2

    # DecoderLayer Testing
    self_attn = src_attn = MultiHeadedAttention(head=numHeads, embedding_dim=C, dropout=dr)
    ff = PositionwiseFeedForward(d_model=C, d_ff=C//4, dropout=dr)
    x = torch.randn(B, L, C)
    memory = torch.randn(B, L, C)
    mask = torch.randn(numHeads, L, L)
    source_mask = target_mask = mask
    decoderLayer = DecoderLayer(d_model=C, self_attn=self_attn, src_attn=src_attn, feed_forward=ff, dropout=dr)
    print('Before DecoderLayer: \t x.shape: ', x.shape)
    print('After DecoderLayer: \t x.shape: ', decoderLayer(x, memory, source_mask, target_mask).shape)
    print('-' * 50)
    
    # Decoder Testing
    x = torch.randn(B, L, C)
    memory = torch.randn(B, L, C)
    mask = torch.randn(numHeads, L, L)
    self_attn = MultiHeadedAttention(head=numHeads, embedding_dim=C, dropout=dr)
    src_attn = MultiHeadedAttention(head=numHeads, embedding_dim=C, dropout=dr)
    ff = PositionwiseFeedForward(d_model=C, d_ff=C//4, dropout=dr)
    decoderLayer = DecoderLayer(d_model=C, self_attn=self_attn, src_attn=src_attn, feed_forward=ff, dropout=dr)
    decoder = Decoder(decoderLayer, 8)
    de_result = decoder(x, memory, source_mask=mask, target_mask=mask)
    print('Before DecoderLayer: \t x.shape: ', x.shape)
    print('After DecoderLayer: \t x.shape: ', de_result.shape)
    print('-' * 50)

    # Generator Testing
    generator = Generator(d_model=C, vocab_size=1000)
    ge_result = generator(de_result)
    print('Before Generator: \t x.shape: ', x.shape)
    print('After Generator: \t x.shape: ', ge_result.shape)
    print('-' * 50)
