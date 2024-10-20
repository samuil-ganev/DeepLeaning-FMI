from parameters import *
from attention import *


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_cross, max_len=750):
        super().__init__()

        pe = torch.zeros(1, max_len, d_cross).to(device)

        position = torch.arange(max_len).unsqueeze(0).unsqueeze(2)
        div_term = (10000.0 ** (torch.arange(0, d_cross, 2) / d_cross)).unsqueeze(0).unsqueeze(0)
        pe[0, :, 0::2] = torch.sin(position / div_term)
        pe[0, :, 1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        # x.shape = (batch, seq_len, d_cross)

        return x


class CLIPEmbedding(nn.Module):
    def __init__(self, n_tokens, d_cross, max_seq_len=77):
        super().__init__()

        # !Забележка! d_cross е размерността на вектора за ембединг на текст - ползвам това име заради attention block-а
        self.token_embedding = nn.Embedding(n_tokens, d_cross, device=device)
        self.position_embedding = PositionalEncoding(d_cross)

    def forward(self, tokens):
        # tokens.shape = (batch, seq_len)
        y = self.token_embedding(tokens)
        # y.shape = (batch, seq_len, d_cross)
        y = self.position_embedding(y) # не е сигурно че ще работи, ако не става да се замени със стандартните синуси и косинуси

        return y


class CLIPLayer(nn.Module): # Norm_First = True Encoder
    def __init__(self, n_heads, d_cross):
        super().__init__()

        self.attn = SelfAttention(n_heads, d_cross)
        self.layer_norm_1 = nn.LayerNorm(d_cross, device=device)
        self.layer_norm_2 = nn.LayerNorm(d_cross, device=device)
        self.W1 = nn.Linear(d_cross, 4 * d_cross, device=device)
        self.W2 = nn.Linear(4 * d_cross, d_cross, device=device)

    def forward(self, y):
        # y.shape = (batch, seq_len, d_cross)
        res = y
        y = self.layer_norm_1(y)

        y = self.attn(y, mask=False) # !да се провери внимателно маската нужна ли е
        # y.shape = (batch, seq_len, d_cross)

        y += res
        res = y
        y = self.layer_norm_2(y)

        y = self.W1(y)
        # y.shape = (batch, seq_len, 4 * d_cross)

        y = F.gelu(y)

        y = self.W2(y)
        # y.shape = (batch, seq_len, d_cross)

        y += res

        return y


class CLIP(nn.Module):
    def __init__(self, token2ind, n_heads=clip_heads_val, d_cross=d_cross_val, num_layers=clip_layers_val):
        super().__init__()

        self.token2ind = token2ind

        self.E = CLIPEmbedding(len(token2ind), d_cross) # len(token2ind) = 49408

        self.layers = nn.ModuleList([
            CLIPLayer(n_heads, d_cross) for _ in range(num_layers)
        ]) # 'num_layers' layers с по 'num_heads' heads

        self.layer_norm = nn.LayerNorm(d_cross, device=device)

    def preparePaddedBatch(self, sents):
        # sents.shape = (batch, seq_len)
        device = next(self.parameters()).device

        m = max(len(s) for s in sents)
        sents = [ [ self.token2ind.get(token, self.token2ind.get(unkToken)) for token in s ] \
                 for s in sents ]
        tokens_padded = [ s+(m-len(s))*[self.token2ind.get(padToken)] for s in sents ]
        # tokens_padded.shape = (batch, seq_len)

        return torch.tensor(tokens_padded, dtype=torch.long, device=device)

    def forward(self, tokens):
        # tokens.shape = (batch, seq_len)
        tokens = self.preparePaddedBatch(tokens)
        tokens = tokens.type(torch.long)

        prompt = self.E(tokens)
        # prompt.shape = (batch, seq_len, d_cross)

        for layer in self.layers:
            prompt = layer(prompt)

        prompt = self.layer_norm(prompt)
        # prompt.shape = (batch, seq_len, d_cross)

        return prompt.to(device)