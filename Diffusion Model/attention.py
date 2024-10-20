from parameters import *


class MultiHeadAttn(nn.Module):
    def __init__(self, n_heads, d_model, d_cross = None):
        super().__init__()

        assert d_model % n_heads == 0
        self.n_heads = n_heads
        if d_cross is None: d_cross = d_model

        self.d_keys = self.d_values = d_model // n_heads

        self.Wq_net = nn.Linear(d_model, n_heads * self.d_keys, device=device)
        self.Wk_net = nn.Linear(d_cross, n_heads * self.d_keys, device=device)
        self.Wv_net = nn.Linear(d_cross, n_heads * self.d_values, device=device)
        self.Wo_net = nn.Linear(n_heads * self.d_values, d_model, device=device)

    def forward(self, x, y = None, mask = False):

        if y is None: y = x
        # (image)   x.shape = (batch, seq_len_x, d_model)   !Забележка! seq_len_x е броят пиксели на картинка - Width * Height, а d_model броят канали
        # (context) y.shape - (batch, seq_len_y, d_cross)
        batch, seq_len_x, d_model = x.shape
        _, seq_len_y, d_cross = y.shape

        view_shape_x = (batch, seq_len_x, self.n_heads, self.d_keys)
        view_shape_y = (batch, seq_len_y, self.n_heads, self.d_keys)

        q = self.Wq_net(x).view(view_shape_x).transpose(1, 2)
        k = self.Wk_net(y).view(view_shape_y).transpose(1, 2)
        v = self.Wv_net(y).view(view_shape_y).transpose(1, 2)
        # q.shape = k.shape = v.shape = (batch, n_heads, seq_len_x(/y), d_keys(=d_values))

        attn = q @ k.transpose(-1, -2)
        # attn.shape = (batch, n_heads, seq_len_x, seq_len_y)

        if mask:
            mask = torch.ones_like(attn, dtype = torch.bool).triu(1).to(device)
            attn.masked_fill_(mask, -torch.inf)

        attn /= (self.d_keys ** 0.5)
        attn = F.softmax(attn, dim=-1)

        attn = attn @ v
        # attn.shape = (batch, n_heads, seq_len_x, d_values(=d_keys))

        attn = attn.transpose(1, 2).flatten(2, 3)
        # attn.shape = (batch, seq_len_x, n_heads * d_values (=d_model))

        attn = self.Wo_net(attn)
        # attn.shape = (batch, seq_len_x, d_model)

        return attn


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_model, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.QKV = nn.Linear(d_model, 3 * d_model, bias=in_proj_bias, device=device)

        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias, device=device)
        self.n_heads = n_heads
        self.d_keys = self.d_values = d_model // n_heads

    def forward(self, x, mask=False):
        # (image) x.shape = (batch, seq_len, d_model)

        b, seq_len, d_model = x.shape

        interim_shape = (b, seq_len, self.n_heads, self.d_keys)

        q, k, v = self.QKV(x).chunk(3, dim=-1)
        # q.shape = k.shape = v.shape = (batch, seq_len, d_model)

        q = q.view(interim_shape).to(device).transpose(1, 2)
        k = k.view(interim_shape).to(device).transpose(1, 2)
        v = v.view(interim_shape).to(device).transpose(1, 2)
        # q.shape = k.shape = v.shape = (batch, n_heads, seq_len_x(/y), d_keys(=d_values))

        attn = q @ k.transpose(-1, -2)
        # attn.shape = (batch, n_heads, seq_len, seq_len)

        if mask:
            mask = torch.ones_like(attn, dtype=torch.bool).triu(1).to(device)
            attn.masked_fill_(mask, -torch.inf)

        attn /= math.sqrt(self.d_keys)
        attn = F.softmax(attn, dim=-1)

        attn = attn @ v
        # attn.shape = (batch, n_heads, seq_len, d_values(=d_keys))

        attn = attn.transpose(1, 2).flatten(2, 3)
        # attn.shape = (batch, seq_len_x, n_heads * d_values (=d_model))

        attn = self.out_proj(attn)
        # attn.shape = (batch, seq_len_x, d_model)

        return attn


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.q_proj = nn.Linear(d_model, d_model, bias=in_proj_bias, device=device)
        self.k_proj = nn.Linear(d_cross, d_model, bias=in_proj_bias, device=device)
        self.v_proj = nn.Linear(d_cross, d_model, bias=in_proj_bias, device=device)

        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias, device=device)
        self.n_heads = n_heads
        self.d_keys = self.d_values = d_model // n_heads

    def forward(self, x, y):
        # (image)   x.shape = (batch, seq_len_x, d_model)   !Забележка! seq_len_x е броят пиксели на картинка - Width * Height, а d_model броят канали
        # (context) y.shape - (batch, seq_len_y, d_cross)

        b, seq_len_x, d_model = x.shape
        interim_shape = (b, -1, self.n_heads, self.d_keys)

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        # q.shape =           (batch, seq_len_x, d_model)
        # k.shape = v.shape = (batch, seq_len_y, d_model)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        # q.shape =           (batch, n_heads, seq_len_x, d_keys)
        # k.shape = v.shape = (batch, n_heads, seq_len_y, d_keys(=d_values))

        attn = q @ k.transpose(-1, -2)
        # attn.shape = (batch, n_heads, seq_len_x, seq_len_y)

        attn /= math.sqrt(self.d_keys)
        attn = F.softmax(attn, dim=-1)

        attn = attn @ v
        # attn.shape = (batch, n_heads, seq_len_x, d_values(=d_keys))

        attn = attn.transpose(1, 2).contiguous().flatten(2, 3)
        # attn.shape = (batch, seq_len_x, n_heads * d_values (=d_model))

        attn = self.out_proj(attn)
        # attn.shape = (batch, seq_len_x, d_model)

        return attn