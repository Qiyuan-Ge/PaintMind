import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, dims, channels=3, groups=16, init_kernel_size=5):
        super().__init__()
        dim_pairs = zip(dims[:-1], dims[1:])

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, dims[0], init_kernel_size, padding=init_kernel_size//2), 
                nn.LeakyReLU(0.1),
            )]
        )

        for dim_in, dim_out in dim_pairs:
            self.layers.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 4, stride=2, padding=1),
                nn.GroupNorm(groups, dim_out),
                nn.LeakyReLU(0.1),
            ))

        dim = dims[-1]
        self.to_logits = nn.Sequential( # return 5 x 5, for PatchGAN-esque training
            nn.Conv2d(dim, dim, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim, 1, 4)
        )

    def forward(self, x):
        for net in self.layers:
            x = net(x)

        return self.to_logits(x)