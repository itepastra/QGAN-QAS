#!/usr/bin/env python

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


import torchvision

SIDE_LENGTH = 3
ITERATIONS = 50000
UPDATE_ITERS = 100
IMAGE_ITERS = 20000
BATCH_SIZE = 30
GENERATOR_INPUT_SIZE = 9

D_STEPS = 1
G_STEPS = 1

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

    def forward(self, x):
        x = self.arch(x)

        x = x.view(-1, SIDE_LENGTH, SIDE_LENGTH)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.arch = nn.Sequential(
            nn.Linear(GENERATOR_INPUT_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, GRID_SIZE),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, GRID_SIZE)
        x = self.arch(x)

        x = x.view(-1, 1)
        return x


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

        self.data = torch.tensor(
            np.vstack((bars[:-1], stripes[1:])), dtype=torch.float32
        ).view((-1, side_length, side_length))

    def __len__(self):
        return 2**self.side_length * 2 - 2

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample, 1


def show_grid(x, y, title=""):
    _, ax = plt.subplots(2)
    ax[0].imshow(x.detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    ax[1].imshow(y.detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    plt.show()


def straight_through_sample(probs):
    hard = (probs > torch.rand_like(probs)).float()
    return hard + (probs - hard).detach()


def main():
    device = "cpu"
    x = torch.randn((1), device=device)
    real_data = BarsAndStripesDataset(SIDE_LENGTH)

    train_loader = torch.utils.data.DataLoader(
        real_data, shuffle=True, batch_size=BATCH_SIZE
    )

    generator = Generator()
    discriminator = Discriminator()
    criterion = nn.BCELoss()

    g_losses = []
    d_losses = []

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-2)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-2)

    for epoch in range(ITERATIONS):
        for i, data in enumerate(train_loader):
            # --------------------
            # Discriminator step
            # --------------------
            r_data, _ = data
            bs = r_data.size(0)
            r_data = r_data.to(device)

            z = torch.randn((bs, GENERATOR_INPUT_SIZE), device=device)
            with torch.no_grad():
                g_logits = generator(z)

            discriminator.zero_grad()
            d_real = discriminator(r_data)
            d_fake = discriminator(g_logits)

            d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(
                d_fake, torch.zeros_like(d_fake)
            )
            d_losses.append(float(d_loss))
            d_loss.backward()
            d_optimizer.step()
            # --------------------
            # Generator step
            # --------------------
            z = torch.randn((bs, GENERATOR_INPUT_SIZE), device=device)

            generator.zero_grad()
            g_probs = generator(z)
            d_fake_for_g = discriminator(g_probs)
            g_loss = criterion(d_fake_for_g, torch.ones_like(d_fake_for_g))
            g_losses.append(float(g_loss))
            g_loss.backward()
            g_optimizer.step()

            if i == len(train_loader) - 1 and (epoch + 1) % UPDATE_ITERS == 0:
                # print("g grads")
                # for name, param in generator.named_parameters():
                #     if param.grad is None:
                #         print(f"No gradient for {name}")
                #     elif param.grad.abs().sum() == 0:
                #         print(f"Zero gradient for {name}")
                #     else:
                #         print(param.grad)
                # print("d grads")
                # for name, param in discriminator.named_parameters():
                #     if param.grad is None:
                #         print(f"No gradient for {name}")
                #     elif param.grad.abs().sum() == 0:
                #         print(f"Zero gradient for {name}")
                #     else:
                #         print(param.grad)
                print(
                    f"Epoch {epoch}: Loss_D: {d_loss.item()}, Loss_G: {g_loss.item()}"
                )
            if i == len(train_loader) - 1 and (epoch + 1) % IMAGE_ITERS == 0:
                with torch.no_grad():
                    g_probs_vis = generator(
                        torch.randn((1, GENERATOR_INPUT_SIZE), device=device)
                    )[0]
                    g_hard_vis = (g_probs_vis > 0.5).float()
                fig, axs = plt.subplots(4, 4)

                for row in axs:
                    for ax in row:
                        g_probs_vis = (
                            generator(
                                torch.randn((1, GENERATOR_INPUT_SIZE), device=device)
                            )
                        )[0]
                        g_hard_vis = (g_probs_vis > 0.5).float()
                        ax.imshow(
                            g_probs_vis.detach().cpu().numpy(),
                            cmap="gray",
                            vmin=0,
                            vmax=1,
                        )
                plt.show()

    fig, axs = plt.subplots(4, 4)

    for row in axs:
        for ax in row:
            g_probs_vis = (
                generator(torch.randn((1, GENERATOR_INPUT_SIZE), device=device))
            )[0]
            g_hard_vis = (g_probs_vis > 0.5).float()
            ax.imshow(g_hard_vis.detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)

    plt.figure()
    average_over = 10
    plt.plot(
        np.convolve(d_losses, np.ones(average_over) / average_over, mode="valid"),
        label="discriminator",
    )
    plt.plot(
        np.convolve(g_losses, np.ones(average_over) / average_over, mode="valid"),
        label="generator",
    )
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
