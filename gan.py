#!/usr/bin/env python

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


import torchvision

SIDE_LENGTH = 4
ITERATIONS = 5000
UPDATE_ITERS = 100
IMAGE_ITERS = 2500
BATCH_SIZE = 16
GENERATOR_INPUT_SIZE = 16

D_STEPS = 1
G_STEPS = 1

GRID_SIZE = SIDE_LENGTH**2


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(GENERATOR_INPUT_SIZE, 32)
        self.non_linear_1 = nn.Tanh()
        self.inner_layer = nn.Linear(32, 32)
        self.non_linear_2 = nn.Tanh()
        self.inner_layer_2 = nn.Linear(32, 32)
        self.non_linear_3 = nn.Tanh()
        self.output_layer = nn.Linear(32, GRID_SIZE)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.non_linear_1(x)
        x = self.inner_layer(x)
        x = self.non_linear_2(x)
        x = self.inner_layer_2(x)
        x = self.non_linear_3(x)
        x = self.output_layer(x)

        x = x.view(-1, SIDE_LENGTH, SIDE_LENGTH)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(GRID_SIZE, 64)
        self.non_linear_1 = nn.LeakyReLU(0.2)
        self.inner_layer = nn.Linear(64, 32)
        self.non_linear_2 = nn.LeakyReLU(0.2)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(-1, GRID_SIZE)
        x = self.input_layer(x)
        x = self.non_linear_1(x)
        x = self.inner_layer(x)
        x = self.non_linear_2(x)
        x = self.output_layer(x)

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
    criterion = nn.BCEWithLogitsLoss()

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)
    )

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
                g_probs = torch.sigmoid(g_logits)  # match real range [0,1]
                g_st = straight_through_sample(g_probs)

            real_labels = torch.ones((bs, 1), device=device)
            fake_labels = torch.zeros((bs, 1), device=device)

            d_optimizer.zero_grad()
            d_real = discriminator(r_data)
            d_fake = discriminator(g_probs)

            # grad = torch.autograd.grad(d_real.sum(), r_data, create_graph=True)[0]
            # r1 = grad.pow(2).view(bs, -1).sum(1).mean()

            d_loss = torch.relu(1 - d_real).mean() + torch.relu(1 + d_fake).mean()
            d_loss.backward()
            d_optimizer.step()
            # --------------------
            # Generator step
            # --------------------
            z = torch.randn((bs, GENERATOR_INPUT_SIZE), device=device)

            g_optimizer.zero_grad()
            g_probs = torch.sigmoid(generator(z))
            d_fake_for_g = discriminator(g_probs)
            binarize = (g_probs * (1 - g_probs)).mean()
            g_loss = -d_fake_for_g.mean()  # + 0.01 * binarize
            g_loss.backward()
            g_optimizer.step()

            if i == len(train_loader) - 1 and (epoch + 1) % UPDATE_ITERS == 0:
                print("g grads")
                for name, param in generator.named_parameters():
                    if param.grad is None:
                        print(f"No gradient for {name}")
                    elif param.grad.abs().sum() == 0:
                        print(f"Zero gradient for {name}")
                    else:
                        print(param.grad)
                print("d grads")
                for name, param in discriminator.named_parameters():
                    if param.grad is None:
                        print(f"No gradient for {name}")
                    elif param.grad.abs().sum() == 0:
                        print(f"Zero gradient for {name}")
                    else:
                        print(param.grad)
                print(
                    f"Epoch {epoch}: Loss_D: {d_loss.item()}, Loss_G: {g_loss.item()}"
                )
            if i == len(train_loader) - 1 and (epoch + 1) % IMAGE_ITERS == 0:
                with torch.no_grad():
                    g_probs_vis = torch.sigmoid(
                        generator(torch.randn((1, GENERATOR_INPUT_SIZE), device=device))
                    )[0]
                    g_hard_vis = (g_probs_vis > 0.5).float()
                show_grid(g_probs_vis, g_hard_vis, f"Epoch {epoch + 1} probs")

    fig, axs = plt.subplots(4, 4)

    for row in axs:
        for ax in row:
            g_probs_vis = torch.sigmoid(
                generator(torch.randn((1, GENERATOR_INPUT_SIZE), device=device))
            )[0]
            g_hard_vis = (g_probs_vis > 0.5).float()
            ax.imshow(g_hard_vis.detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    plt.show()


if __name__ == "__main__":
    main()
