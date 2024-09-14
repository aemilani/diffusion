import os
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay."""
    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def create_mnist_dataloaders(batch_size, image_size=28, num_workers=4):
    preprocess = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])])  # [0,1] to [-1,1]
    train_dataset = MNIST(root="../data", train=True, download=True, transform=preprocess)
    test_dataset = MNIST(root="../data", train=False, download=True, transform=preprocess)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers), \
        DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n, self.groups, c // self.groups, h, w)  # group
        x = x.transpose(1, 2).contiguous().view(n, -1, h, w)  # shuffle
        return x


class ConvBnSiLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.module = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                                    nn.BatchNorm2d(out_channels), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.module(x)


class ResidualBottleneck(nn.Module):
    """ShuffleNet_v2 basic unit(https://arxiv.org/pdf/1807.11164.pdf)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1,
                                               groups=in_channels // 2), nn.BatchNorm2d(in_channels // 2),
                                     ConvBnSiLu(in_channels // 2, out_channels // 2, 1, 1, 0))
        self.branch2 = nn.Sequential(ConvBnSiLu(in_channels // 2, in_channels // 2, 1, 1, 0),
                                     nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1,
                                               groups=in_channels // 2), nn.BatchNorm2d(in_channels // 2),
                                     ConvBnSiLu(in_channels // 2, out_channels // 2, 1, 1, 0))
        self.channel_shuffle = ChannelShuffle(2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x = torch.cat([self.branch1(x1), self.branch2(x2)], dim=1)
        x = self.channel_shuffle(x)  # shuffle two branches
        return x


class ResidualDownsample(nn.Module):
    """ShuffleNet_v2 unit for spatial down sampling(https://arxiv.org/pdf/1807.11164.pdf)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 2, 1,
                                               groups=in_channels), nn.BatchNorm2d(in_channels),
                                     ConvBnSiLu(in_channels, out_channels // 2, 1, 1, 0))
        self.branch2 = nn.Sequential(ConvBnSiLu(in_channels, out_channels // 2, 1, 1, 0),
                                     nn.Conv2d(out_channels // 2, out_channels // 2, 3, 2, 1,
                                               groups=out_channels // 2), nn.BatchNorm2d(out_channels // 2),
                                     ConvBnSiLu(out_channels // 2, out_channels // 2, 1, 1, 0))
        self.channel_shuffle = ChannelShuffle(2)

    def forward(self, x):
        x = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        x = self.channel_shuffle(x)  # shuffle two branches
        return x


class TimeMLP(nn.Module):
    """Introduction of timestep information to feature maps with mlp."""
    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                 nn.SiLU(),
                                 nn.Linear(hidden_dim, out_dim))
        self.act = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb
        return self.act(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.conv0 = nn.Sequential(*[ResidualBottleneck(in_channels, in_channels) for i in range(3)],
                                   ResidualBottleneck(in_channels, out_channels // 2))
        self.time_mlp = TimeMLP(embedding_dim=time_embedding_dim, hidden_dim=out_channels, out_dim=out_channels // 2)
        self.conv1 = ResidualDownsample(out_channels // 2, out_channels)

    def forward(self, x, t=None):
        x_shortcut = self.conv0(x)
        if t is not None:
            x = self.time_mlp(x_shortcut, t)
        x = self.conv1(x)
        return [x, x_shortcut]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv0 = nn.Sequential(*[ResidualBottleneck(in_channels, in_channels) for i in range(3)],
                                   ResidualBottleneck(in_channels, in_channels // 2))
        self.time_mlp = TimeMLP(embedding_dim=time_embedding_dim, hidden_dim=in_channels, out_dim=in_channels // 2)
        self.conv1 = ResidualBottleneck(in_channels // 2, out_channels // 2)

    def forward(self, x, x_shortcut, t=None):
        x = self.upsample(x)
        x = torch.cat([x, x_shortcut], dim=1)
        x = self.conv0(x)
        if t is not None:
            x = self.time_mlp(x, t)
        x = self.conv1(x)
        return x


class Unet(nn.Module):
    """Simple U-Net architecture without attention"""
    def __init__(self, timesteps, time_embedding_dim, in_channels=3, out_channels=2, base_dim=32,
                 dim_mults=[2, 4, 8, 16]):
        super().__init__()
        assert isinstance(dim_mults, (list, tuple))
        assert base_dim % 2 == 0

        channels = self._cal_channels(base_dim, dim_mults)

        self.init_conv = ConvBnSiLu(in_channels, base_dim, 3, 1, 1)
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(c[0], c[1], time_embedding_dim) for c in channels])
        self.decoder_blocks = nn.ModuleList([DecoderBlock(c[1], c[0], time_embedding_dim) for c in channels[::-1]])

        self.mid_block = nn.Sequential(*[ResidualBottleneck(channels[-1][1], channels[-1][1]) for i in range(2)],
                                       ResidualBottleneck(channels[-1][1], channels[-1][1] // 2))

        self.final_conv = nn.Conv2d(in_channels=channels[0][0] // 2, out_channels=out_channels, kernel_size=1)

    def forward(self, x, t=None):
        x = self.init_conv(x)
        if t is not None:
            t = self.time_embedding(t)
        encoder_shortcuts = []
        for encoder_block in self.encoder_blocks:
            x, x_shortcut = encoder_block(x, t)
            encoder_shortcuts.append(x_shortcut)
        x = self.mid_block(x)
        encoder_shortcuts.reverse()
        for decoder_block, shortcut in zip(self.decoder_blocks, encoder_shortcuts):
            x = decoder_block(x, shortcut, t)
        x = self.final_conv(x)
        return x

    def _cal_channels(self, base_dim, dim_mults):
        dims = [base_dim * x for x in dim_mults]
        dims.insert(0, base_dim)
        channels = []
        for i in range(len(dims) - 1):
            channels.append((dims[i], dims[i + 1]))  # in_channel, out_channel
        return channels


class MNISTDiffusion(nn.Module):
    def __init__(self, image_size, in_channels, time_embedding_dim=256, timesteps=1000, base_dim=32,
                 dim_mults=[1, 2, 4, 8]):
        super().__init__()
        self.timesteps = timesteps
        self.in_channels = in_channels
        self.image_size = image_size

        betas = self._cosine_variance_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=-1)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1. - alphas_cumprod))

        self.model = Unet(timesteps, time_embedding_dim, in_channels, in_channels, base_dim, dim_mults)

    def forward(self, x, noise):
        # x:NCHW
        t = torch.randint(0, self.timesteps, (x.shape[0],)).to(x.device)
        x_t = self._forward_diffusion(x, t, noise)
        pred_noise = self.model(x_t, t)
        return pred_noise

    @torch.no_grad()
    def sampling(self, n_samples, clipped_reverse_diffusion=True, device="cpu"):
        x_t = torch.randn((n_samples, self.in_channels, self.image_size, self.image_size)).to(device)
        for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling"):
            noise = torch.randn_like(x_t).to(device)
            t = torch.tensor([i for _ in range(n_samples)]).to(device)
            if clipped_reverse_diffusion:
                x_t = self._reverse_diffusion_with_clip(x_t, t, noise)
            else:
                x_t = self._reverse_diffusion(x_t, t, noise)
        x_t = (x_t + 1.) / 2.  # [-1,1] to [0,1]
        return x_t

    def sample_intermediate_images(self, freq=100, clipped_reverse_diffusion=True, device="cpu"):
        images = []
        x_t = torch.randn((1, self.in_channels, self.image_size, self.image_size)).to(device)
        for i in tqdm(range(self.timesteps - 1, -1, -1), desc="Sampling"):
            noise = torch.randn_like(x_t).to(device)
            t = torch.tensor(i).to(device)
            if clipped_reverse_diffusion:
                x_t = self._reverse_diffusion_with_clip(x_t, t, noise)
            else:
                x_t = self._reverse_diffusion(x_t, t, noise)
            if i % freq == 0:
                images.append((x_t + 1.) / 2.)
        return images

    def _cosine_variance_schedule(self, timesteps, epsilon=0.008):
        steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5) ** 2
        betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
        return betas

    def _forward_diffusion(self, x_0, t, noise):
        assert x_0.shape == noise.shape
        # q(x_{t}|x_{t-1})
        return self.sqrt_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * x_0 + \
            self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_0.shape[0], 1, 1, 1) * noise

    @torch.no_grad()
    def _reverse_diffusion(self, x_t, t, noise):
        pred = self.model(x_t, t)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        mean = (1. / torch.sqrt(alpha_t)) * (x_t - ((1.0 - alpha_t) / sqrt_one_minus_alpha_cumprod_t) * pred)
        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1, 1)
            std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
        else:
            std = 0.0
        return mean + std * noise

    @torch.no_grad()
    def _reverse_diffusion_with_clip(self, x_t, t, noise):
        pred = self.model(x_t, t)
        alpha_t = self.alphas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        alpha_t_cumprod = self.alphas_cumprod.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(x_t.shape[0], 1, 1, 1)
        x_0_pred = torch.sqrt(1. / alpha_t_cumprod) * x_t - torch.sqrt(1. / alpha_t_cumprod - 1.) * pred
        x_0_pred.clamp_(-1., 1.)
        if t.min() > 0:
            alpha_t_cumprod_prev = self.alphas_cumprod.gather(-1, t - 1).reshape(x_t.shape[0], 1, 1, 1)
            mean = (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod)) * x_0_pred + \
                   ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod)) * x_t
            std = torch.sqrt(beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))
        else:
            mean = (beta_t / (1. - alpha_t_cumprod)) * x_0_pred  # alpha_t_cumprod_prev=1 since 0!=1
            std = 0.0
        return mean + std * noise


if __name__ == "__main__":
    lr = 0.001
    batch_size = 128
    epochs = 100
    ckpt = ''
    n_samples = 9
    model_base_dim = 16
    timesteps = 1000
    model_ema_steps = 10
    model_ema_decay = 0.995
    log_freq = 10

    device = "cpu"
    train_dataloader, test_dataloader = create_mnist_dataloaders(batch_size=batch_size, image_size=28)
    model = MNISTDiffusion(timesteps=timesteps,
                           image_size=28,
                           in_channels=1,
                           base_dim=model_base_dim,
                           dim_mults=[2, 4]).to(device)

    adjust = 1 * batch_size * model_ema_steps / epochs
    alpha = 1.0 - model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, lr, total_steps=epochs * len(train_dataloader), pct_start=0.25,
                           anneal_strategy='cos')
    loss_fn = nn.MSELoss(reduction='mean')

    # load checkpoint
    if ckpt:
        ckpt = torch.load(ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    # Training
    global_steps = 0
    for i in range(epochs):
        model.train()
        for j, (image, target) in enumerate(train_dataloader):
            noise = torch.randn_like(image).to(device)
            image = image.to(device)
            pred = model(image, noise)
            loss = loss_fn(pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps % model_ema_steps == 0:
                model_ema.update_parameters(model)
            global_steps += 1
            if j % log_freq == 0:
                print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i + 1, epochs, j,
                                                                              len(train_dataloader),
                                                                              loss.detach().cpu().item(),
                                                                              scheduler.get_last_lr()[0]))
        ckpt = {"model": model.state_dict(),
                "model_ema": model_ema.state_dict()}

        os.makedirs("results", exist_ok=True)
        torch.save(ckpt, "results/steps_{:0>8}.pt".format(global_steps))

        model_ema.eval()
        samples = model_ema.module.sampling(n_samples, clipped_reverse_diffusion=True, device=device)
        save_image(samples, "results/steps_{:0>8}.png".format(global_steps), nrow=int(math.sqrt(n_samples)))

    # Plotting the samples saved from the last epoch
    img = mpimg.imread("results/steps_{:0>8}.png".format(global_steps))

    plt.figure(figsize=(2, 2))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Plotting the intermediate diffusion steps
    imgs = model_ema.module.sample_intermediate_images()
    fig, axes = plt.subplots(1, len(imgs), figsize=(12, 2))
    for img, ax in zip(imgs, axes.flat):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('steps.png')
