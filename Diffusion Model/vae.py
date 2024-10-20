from parameters import *
from attention import *


class VAE_Attn(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.gn = nn.GroupNorm(32, channels) # това число 32 да се прегледа в последствие
        self.attn = SelfAttention(1, channels)

    def forward(self, x):
        # x.shape = (batch, c, h, w)
        batch, c, h, w = x.shape

        res = x
        x = self.gn(x)

        x = x.view(batch, c, h * w).transpose(-1, -2)
        # x.shape = (batch, h * w, c)

        x = self.attn(x)
        # x.shape = (batch, h * w, c)

        x = x.transpose(-1, -2).view(batch, c, h, w)
        # x.shape = (batch, c, h, w)

        x += res

        return x


class VAE_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.gn1 = nn.GroupNorm(32, in_channels) # това число 32 да се прегледа в последствие
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.gn2 = nn.GroupNorm(32, out_channels) # това число 32 да се прегледа в последствие
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)


        if in_channels == out_channels:
            self.res_layer = nn.Identity()
        else:
            self.res_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # x.shape = (batch, in_channels, h, w)

        res = x

        x = self.gn1(x)
        # x.shape = (batch, in_channels, h, w)

        x = F.silu(x)

        x = self.conv1(x)
        # x.shape = (batch, out_channels, h, w)

        x = self.gn2(x)
        x = F.silu(x)

        x = self.conv2(x)
        # x.shape = (batch, out_channels, h, w)

        return x + self.res_layer(res)


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # x.shape = (batch, c, h, w)

            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # x.shape = (batch, 128, h, w)

            VAE_ResBlock(128, 128),
            # x.shape = (batch, 128, h, w)

            VAE_ResBlock(128, 128),
            # x.shape = (batch, 128, h, w)

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # x.shape = (batch, 128, h / 2, w / 2)

            VAE_ResBlock(128, 256),
            # x.shape = (batch, 256, h / 2, w / 2)

            VAE_ResBlock(256, 256),
            # x.shape = (batch, 256, h / 2, w / 2)

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # x.shape = (batch, 256, h / 4, w / 4)

            VAE_ResBlock(256, 512),
            # x.shape = (batch, 512, h / 4, w / 4)

            VAE_ResBlock(512, 512),
            # x.shape = (batch, 512, h / 4, w / 4)

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # x.shape = (batch, 512, h / 8, w / 8)

            VAE_ResBlock(512, 512),
            # x.shape = (batch, 512, h / 8, w / 8)

            VAE_ResBlock(512, 512),
            # x.shape = (batch, 512, h / 8, w / 8)

            VAE_ResBlock(512, 512),
            # x.shape = (batch, 512, h / 8, w / 8)

            VAE_Attn(512),

            VAE_ResBlock(512, 512),
            # x.shape = (batch, 512, h / 8, w / 8)

            nn.GroupNorm(32, 512), # това число 32 да се прегледа в последствие

            nn.SiLU(),

            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # x.shape = (batch, 8, h / 8, w / 8)

            nn.Conv2d(8, 8, kernel_size=1, padding=0)
            # x.shape = (batch, 8, h / 8, w / 8)
        )

    def forward(self, x):
        # x.shape = (batch, c, h, w)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        # x.shape = (batch, 8, h / 8, w / 8)

        mean, log_var = torch.chunk(x, 2, dim=1)
        # mean.shape = log_var.shape = (batch, 4, h / 8, w / 8)

        log_var = torch.clamp(log_var, -30, 20)
        var = log_var.exp()
        st_dev = var.sqrt()

        noise = torch.randn(mean.shape).to(device)
        x = mean + st_dev * noise

        x *= 0.18215

        return x


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # x.shape = (batch, 4, h / 8, w / 8)

            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            # x.shape = (batch, 4, h / 8, w / 8)

            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            # x.shape = (batch, 512, h / 8, w / 8)

            VAE_ResBlock(512, 512),
            # x.shape = (batch, 512, h / 8, w / 8)

            VAE_Attn(512),
            # x.shape = (batch, 512, h / 8, w / 8)

            VAE_ResBlock(512, 512),
            # x.shape = (batch, 512, h / 8, w / 8)

            VAE_ResBlock(512, 512),
            # x.shape = (batch, 512, h / 8, w / 8)

            VAE_ResBlock(512, 512),
            # x.shape = (batch, 512, h / 8, w / 8)

            VAE_ResBlock(512, 512),
            # x.shape = (batch, 512, h / 8, w / 8)

            nn.Upsample(scale_factor=2),
            # x.shape = (batch, 512, h / 4, w / 4)

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # x.shape = (batch, 512, h / 4, w / 4)

            VAE_ResBlock(512, 512),
            # x.shape = (batch, 512, h / 4, w / 4)

            VAE_ResBlock(512, 512),
            # x.shape = (batch, 512, h / 4, w / 4)

            VAE_ResBlock(512, 512),
            # x.shape = (batch, 512, h / 4, w / 4)

            nn.Upsample(scale_factor=2),
            # x.shape = (batch, 512, h / 2, w / 2)

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # x.shape = (batch, 512, h / 2, w / 2)

            VAE_ResBlock(512, 256),
            # x.shape = (batch, 256, h / 2, w / 2)

            VAE_ResBlock(256, 256),
            # x.shape = (batch, 256, h / 2, w / 2)

            VAE_ResBlock(256, 256),
            # x.shape = (batch, 256, h / 2, w / 2)

            nn.Upsample(scale_factor=2),
            # x.shape = (batch, 256, h, w)

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # x.shape = (batch, 256, h, w)

            VAE_ResBlock(256, 128),
            # x.shape = (batch, 128, h, w)

            VAE_ResBlock(128, 128),
            # x.shape = (batch, 128, h, w)

            VAE_ResBlock(128, 128),
            # x.shape = (batch, 128, h, w)

            nn.GroupNorm(32, 128),
            # x.shape = (batch, 128, h, w)

            nn.SiLU(),

            nn.Conv2d(128, 3, kernel_size=3, padding=1)
            # x.shape = (batch, 3, h, w)
    )

    def forward(self, x):
        # x.shape = (batch, 4, h / 4, w / 4)

        x /= 0.18215

        for module in self:
            x = module(x)
        # x.shape = (batch, 3, h, w)

        return x


