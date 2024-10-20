from parameters import *
from attention import *
from PIL import Image
from clip import *


class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.W1 = nn.Linear(d_model, 4 * d_model, device=device)
        self.W2 = nn.Linear(4 * d_model, 4 * d_model, device=device)

    def forward(self, x):
        # x.shape = (1, 320)

        x = self.W1(x)
        # x.shape = (1, 1280)

        x = F.silu(x)

        x = self.W2(x)
        # x.shape = (1, 1280)

        return x


class UNET_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()

        self.gn1 = nn.GroupNorm(32, in_channels, device=device) # това число 32 да се прегледа в последствие
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, device=device)
        self.W1 = nn.Linear(n_time, out_channels, device=device)

        self.gn2 = nn.GroupNorm(32, out_channels, device=device) # това число 32 да се прегледа в последствие
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, device=device)

        self.res_layer = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, device=device)

    def forward(self, feature, time):
        # feature.shape = (batch, in_channels, h, w)
        # time.shape = (1, 1280)

        res = feature

        feature = self.gn1(feature)
        feature = F.silu(feature)

        feature = self.conv1(feature)
        # feature.shape = (batch, out_channels, h, w)

        time = F.silu(time)
        time = self.W1(time)
        # time.shape = (1, out_channels)

        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        # merged.shape = (batch, out_channels, h, w)

        merged = self.gn2(merged)
        merged = F.silu(merged)

        merged = self.conv2(merged)
        # merged.shape = (batch, out_channels, h, w)

        return merged + self.res_layer(res)


class UNET_Attn(nn.Module):
    def __init__(self, n_heads, d_model, d_cross=d_cross_val):
        super().__init__()

        channels = n_heads * d_model

        self.gn1 = nn.GroupNorm(32, channels, eps=1e-6, device=device) # това число 32 да се прегледа в последствие
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, device=device)

        self.layer_norm_1 = nn.LayerNorm(channels, device=device)
        self.attn1 = SelfAttention(n_heads, channels, in_proj_bias=False)
        self.layer_norm_2 = nn.LayerNorm(channels, device=device)
        self.attn2 = CrossAttention(n_heads, channels, d_cross, in_proj_bias=False)
        self.layer_norm_3 = nn.LayerNorm(channels, device=device)
        self.W1 = nn.Linear(channels, 4 * channels * 2, device=device)
        self.W2 = nn.Linear(4 * channels, channels, device=device)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0, device=device)

    def forward(self, x, y):
        # (image)   x.shape = (batch, c, h, w)
        # (context) y.shape = (batch, seq_len, d_cross)

        res_long = x

        x = self.gn1(x)
        x = self.conv1(x)

        b, c, h, w = x.shape

        x = x.view(b, c, h * w)
        # x.shape = (b, c, h * w)

        x = x.transpose(-1, -2)
        # x.shape = (b, h * w, c)

        res_short = x

        x = self.layer_norm_1(x)
        x = self.attn1(x)
        # x.shape = (b, h * w, c)

        x += res_short
        res_short = x

        x = self.layer_norm_2(x)
        x = self.attn2(x, y)
        # x.shape = (batch, h * w, c)

        x += res_short

        res_short = x

        x = self.layer_norm_3(x)

        x, gate = self.W1(x).chunk(2, dim=-1)
        # x.shape = gate.shape = (batch, h * w, 4 * c)

        x = x * F.gelu(gate)
        x = self.W2(x)
        # x.shape = (batch, h * w, c)

        x += res_short
        x = x.transpose(-1, -2).view(b, c, h, w)
        # x.shape = (batch, c, h, w)

        return self.conv2(x) + res_long


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, device=device)

    def forward(self, x):
        # x.shape = (batch, c, h, w)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # x.shape = (batch, c, h * 2, w * 2)
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(self, x, y, time):
        for layer in self:
            if isinstance(layer, UNET_Attn):
                x = layer(x, y)
            elif isinstance(layer, UNET_ResBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # x.shape = (batch, 4, h / 8, w / 8)

            SwitchSequential(nn.Conv2d(4, 32, kernel_size=3, padding=1, device=device)),
            # x.shape = (batch, 32, h / 8, w / 8)

            SwitchSequential(UNET_ResBlock(32, 32), UNET_Attn(2, 16)),
            # x.shape = (batch, 32, h / 8, w / 8)

            SwitchSequential(UNET_ResBlock(32, 32), UNET_Attn(2, 16)),
            # x.shape = (batch, 32, h / 8, w / 8)

            SwitchSequential(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, device=device)),
            # x.shape = (batch, 32, h / 16, w / 16)

            SwitchSequential(UNET_ResBlock(32, 64), UNET_Attn(2, 32)),
            # x.shape = (batch, 64, h / 16, w / 16)

            SwitchSequential(UNET_ResBlock(64, 64), UNET_Attn(2, 32)),
            # x.shape = (batch, 64, h / 16, w / 16)

            SwitchSequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, device=device)),
            # x.shape = (batch, 64, h / 32, w / 32)

            SwitchSequential(UNET_ResBlock(64, 128), UNET_Attn(4, 32)),
            # x.shape = (batch, 128, h / 32, w / 32)

            SwitchSequential(UNET_ResBlock(128, 128), UNET_Attn(4, 32)),
            # x.shape = (batch, 128, h / 32, w / 32)

            # SwitchSequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, device=device)),
            # x.shape = (batch, 128, h / 64, width / 64)

            # SwitchSequential(UNET_ResBlock(128, 128)),
            # x.shape = (batch, 128, height / 64, width / 64)

            # SwitchSequential(UNET_ResBlock(128, 128))
            # x.shape = (batch, 128, height / 64, width / 64)

        ])

        self.bottleneck = SwitchSequential(
            UNET_ResBlock(128, 128),

            UNET_Attn(4, 32), # 4 * 32 = 128

            UNET_ResBlock(128, 128)
        )

        self.decoders = nn.ModuleList([
            # x.shape = (batch, 256, h / 64, w / 64)

            # SwitchSequential(UNET_ResBlock(256, 128)),
            # x.shape = (batch, 128, h / 64, w / 64)

            # SwitchSequential(UNET_ResBlock(256, 128)),
            # x.shape = (batch, 128, h / 64, w / 64),

            # SwitchSequential(UNET_ResBlock(256, 128), Upsample(128)),
            # x.shape = (batch, 128, h / 32, w / 32)

            SwitchSequential(UNET_ResBlock(256, 128), UNET_Attn(4, 32)),
            # x.shape = (batch, 128, h / 32, w / 32)

            SwitchSequential(UNET_ResBlock(256, 128), UNET_Attn(4, 32)),
            # x.shape = (batch, 128, h / 32, w / 32)

            SwitchSequential(UNET_ResBlock(192, 128), UNET_Attn(2, 64), Upsample(128)),
            # x.shape = (batch, 128, h / 16, w / 16)

            SwitchSequential(UNET_ResBlock(192, 64), UNET_Attn(2, 32)),
            # x.shape = (batch, 64, h / 16, w / 16)

            SwitchSequential(UNET_ResBlock(128, 64), UNET_Attn(2, 32)),
            # x.shape = (batch, 64, h / 16, w / 16)

            SwitchSequential(UNET_ResBlock(96, 64), UNET_Attn(2, 32), Upsample(64)),
            # x.shape = (batch, 64, h / 8, w / 8)

            SwitchSequential(UNET_ResBlock(96, 32), UNET_Attn(2, 16)),
            #x.shape = (batch, 32, h / 8, w / 8)

            SwitchSequential(UNET_ResBlock(64, 32), UNET_Attn(2, 16)),
            # x.shape = (batch, 32, h / 8, w / 8)

            SwitchSequential(UNET_ResBlock(64, 32), UNET_Attn(2, 16))
            # x.shape = (batch, 32, h / 8, w / 8)

        ])

        # Output Layer
        self.gn = nn.GroupNorm(32, 32, device=device) # това число 32 да се прегледа в последствие
        self.conv = nn.Conv2d(32, 4, kernel_size=3, padding=1, device=device)

    def forward(self, x, y, time):
        # (image)   x.shape = (batch, 4, h / 8, w / 8)
        # (context) y.shape = (batch, seq_len, d_cross)
        # time.shape = (1, 1280)

        skip_conns = []
        for layers in self.encoders:
            x = layers(x, y, time)
            skip_conns.append(x)

        x = self.bottleneck(x, y, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_conns.pop()), dim=1)
            x = layers(x, y, time)

        # Output Layer
        x = self.gn(x)
        x = F.silu(x)
        x = self.conv(x)
        # x.shape = (batch, 4, h / 8, w / 8)

        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()

    def forward(self, x, y, time):
        # (image)   x.shape = (batch, 4, h / 8, w / 8)
        # (context) y.shape = (batch, seq_len, d_cross)
        # time.shape = (1, 320)

        time = self.time_embedding(time)
        # time.shape = (1, 1280)

        output = self.unet(x, y, time)
        # output.shape = (batch, 4, h / 8, w / 8)

        return output


class TransformerLayer(nn.Module): # Norm_First = True Decoder
    def __init__(self, n_heads, d_model, d_cross):
        super().__init__()

        self.attn = SelfAttention(n_heads, d_model)
        self.layer_norm_1 = nn.LayerNorm(d_model, device=device)
        self.attn2 = CrossAttention(n_heads, d_model, d_cross)
        self.layer_norm_2 = nn.LayerNorm(d_model, device=device)
        self.layer_norm_3 = nn.LayerNorm(d_model, device=device)
        self.W1 = nn.Linear(d_model, 4 * d_model, device=device)
        self.W2 = nn.Linear(4 * d_model, d_model, device=device)

    def forward(self, x, y):

        res = x
        x = self.layer_norm_1(x)

        x = self.attn(x, mask=False) # !да се провери внимателно маската нужна ли е

        x += res
        res = x
        x = self.layer_norm_2(x)

        x = self.attn2(x, y)
        x += res

        x = self.layer_norm_3(x)
        x = self.W1(x)
        x = F.gelu(x)

        x = self.W2(x)

        x += res

        return x


class VisualTransformer(nn.Module):
    def __init__(self, n_heads, d_model, d_cross, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(n_heads, d_model, d_cross) for _ in range(num_layers)
        ])  # 'num_layers' layers с по 'num_heads' heads
        # self.position_embedding = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model, device=device)

    def split_image_tensor_to_squares(self, image_tensor):
        _, C, H, W = image_tensor.shape

        square_height = H // 4
        square_width = W // 4

        vectors = []

        for i in range(4):
            for j in range(4):
                top = i * square_height
                left = j * square_width
                bottom = (i + 1) * square_height
                right = (j + 1) * square_width

                square = image_tensor[0][:, top:bottom, left:right]
                square_flattened = square.flatten()

                vectors.append(square_flattened)

        tensor = torch.stack(vectors, dim=0)

        return tensor

    def forward(self, x, y):
        # (image)   x.shape = (batch, 4, h / 8, w / 8)
        # (context) y.shape = (batch, seq_len, d_cross)
        x = self.split_image_tensor_to_squares(x)
        x = x.unsqueeze(0)
        # x = self.position_embedding(x)

        for layer in self.layers:
            x = layer(x, y)

        x = self.layer_norm(x)
        return x.view(1, 4, 28, 28).to(device)
