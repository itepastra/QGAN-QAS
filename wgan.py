#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

SIDE_LENGTH = 3
ITERATIONS = 50000
UPDATE_ITERS = 100
IMAGE_ITERS = 10000
BATCH_SIZE = 30
GENERATOR_INPUT_SIZE = 9

N_CRITIC = 1  # critic steps per generator step
LAMBDA_GP = 5.0  # gradient penalty weight

GRID_SIZE = SIDE_LENGTH**2


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(GENERATOR_INPUT_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, GRID_SIZE),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.arch(z)
        return x.view(-1, SIDE_LENGTH, SIDE_LENGTH)


class Critic(nn.Module):
    """WGAN critic: scalar output, no sigmoid."""

    def __init__(self):
        super().__init__()
        self.arch = nn.Sequential(
            nn.Linear(GRID_SIZE, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),  # scalar score
        )

    def forward(self, x):
        x = x.view(-1, GRID_SIZE)
        return self.arch(x)


class BarsAndStripesDataset(Dataset):
    def __init__(self, side_length):
        self.side_length = side_length

        bitstrings = np.array(
            [list(np.binary_repr(i, side_length)) for i in range(2**side_length)],
            dtype=np.float32,
        )

        # stripes are horizontal
        stripes = np.repeat(bitstrings, side_length, 1)

        # bars are vertical
        bars = np.repeat(bitstrings, side_length, 0).reshape(
            (2**side_length, side_length**2)
        )

        fig, axs = plt.subplots(4, 4)
        i = 0
        for row in axs:
            for ax in row:
                if i < len(stripes):
                    ax.imshow(stripes[i].reshape(3, 3), cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(
                        bars[i - len(stripes)].reshape(3, 3),
                        cmap="gray",
                        vmin=0,
                        vmax=1,
                    )
                i += 1
                ax.set_yticks([])
                ax.set_xticks([])
        plt.show()
        # make sure to keep only one fully empty and one full square
        self.data = torch.tensor(
            np.vstack((bars[:-1], stripes[1:])), dtype=torch.float32
        ).view((-1, side_length, side_length))

    def __len__(self):
        return 2**self.side_length * 2 - 2

    def __getitem__(self, idx):
        return self.data[idx], 1


def show_grid(x, y, title=""):
    _, ax = plt.subplots(2)
    ax[0].imshow(x.detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    ax[1].imshow(y.detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    plt.show()


def gradient_penalty(critic, real, fake, device="cpu"):
    bs = real.size(0)
    # interpolate in data space
    eps = torch.rand(bs, 1, 1, device=device)
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)

    scores = critic(x_hat)  # (bs, 1)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # (bs, 3, 3)

    grads = grads.view(bs, -1)
    grad_norm = grads.norm(2, dim=1)
    gp = ((grad_norm - 1.0) ** 2).mean()
    return gp


def main():
    device = "cpu"

    real_data = BarsAndStripesDataset(SIDE_LENGTH)
    train_loader = torch.utils.data.DataLoader(
        real_data, shuffle=True, batch_size=BATCH_SIZE
    )

    g_losses = []
    c_losses = []

    generator = Generator().to(device)
    discriminator = Critic().to(device)

    # WGAN-GP commonly uses these betas
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
    c_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=1e-4, betas=(0.0, 0.9)
    )

    step = 0
    for epoch in range(ITERATIONS):
        for i, (real, _) in enumerate(train_loader):
            real = real.to(device)
            bs = real.size(0)

            # --------------------
            # Critic updates
            # --------------------
            for _ in range(N_CRITIC):
                z = torch.randn(bs, GENERATOR_INPUT_SIZE, device=device)
                c_optimizer.zero_grad()
                fake = generator(z).detach()

                c_real = discriminator(real).mean()
                c_fake = discriminator(fake).mean()

                gp = gradient_penalty(discriminator, real, fake, device=device)
                c_loss = (c_fake - c_real) + LAMBDA_GP * gp

                c_loss.backward()
                c_optimizer.step()

                c_losses.append(float(c_loss))

            # --------------------
            # Generator update
            # --------------------
            g_optimizer.zero_grad()
            z = torch.randn(bs, GENERATOR_INPUT_SIZE, device=device)
            fake = generator(z)
            g_loss = -discriminator(fake).mean()

            g_loss.backward()
            g_optimizer.step()

            g_losses.append(float(g_loss))

            step += 1

            if (step % UPDATE_ITERS) == 0:
                print(
                    f"step {step} | C_loss: {c_loss.item():.4f} | G_loss: {g_loss.item():.4f} | "
                    f"E[C(real)]: {c_real.item():.4f} | E[C(fake)]: {c_fake.item():.4f}"
                )

            if (step % IMAGE_ITERS) == 0:
                with torch.no_grad():
                    fig, axs = plt.subplots(4, 4)
                    for row in axs:
                        for ax in row:
                            sample = generator(
                                torch.randn(1, GENERATOR_INPUT_SIZE, device=device)
                            )[0]
                            ax.imshow(sample.detach().cpu().numpy(), cmap="gray")
                    plt.show()

        # optional early exit for sanity
        # if epoch > 1000: break

    # Final samples (hard thresholded)
    with torch.no_grad():
        fig, axs = plt.subplots(4, 4)
        for row in axs:
            for ax in row:
                sample = generator(torch.randn(1, GENERATOR_INPUT_SIZE, device=device))[
                    0
                ]
                hard = (sample > 0.5).float()
                ax.imshow(hard.detach().cpu().numpy(), cmap="gray")
        plt.show()

    # Loss curves
    plt.figure()
    avg = 50
    if len(c_losses) > avg:
        plt.plot(
            np.linspace(0, len(g_losses), len(c_losses) - avg + 1),
            np.convolve(c_losses, np.ones(avg) / avg, mode="valid"),
            label="critic",
        )
    if len(g_losses) > avg:
        plt.plot(
            np.convolve(g_losses, np.ones(avg) / avg, mode="valid"), label="generator"
        )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
