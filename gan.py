#!/usr/bin/env python

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


import torchvision

SIDE_LENGTH = 3
ITERATIONS = 500
BATCH_SIZE = 1
GENERATOR_INPUT_SIZE = 9

GRID_SIZE = SIDE_LENGTH**2


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(GENERATOR_INPUT_SIZE, 16)
        self.non_linear_1 = nn.ReLU()
        self.inner_layer = nn.Linear(16, 16)
        self.non_linear_2 = nn.ReLU()
        self.output_layer = nn.Linear(16, GRID_SIZE)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.non_linear_1(x)
        x = self.inner_layer(x)
        x = self.non_linear_2(x)
        x = self.output_layer(x)

        x = x.view(-1, SIDE_LENGTH, SIDE_LENGTH)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(GRID_SIZE, 32)
        self.non_linear_1 = nn.ReLU()
        self.inner_layer = nn.Linear(32, 32)
        self.non_linear_2 = nn.ReLU()
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


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    device = "cpu"
    x = torch.randn((1), device=device)
    real_data = BarsAndStripesDataset(3)

    train_loader = torch.utils.data.DataLoader(
        real_data, shuffle=True, batch_size=BATCH_SIZE
    )

    generator = Generator()
    discriminator = Discriminator()
    criterion = nn.BCEWithLogitsLoss()

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

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

            real_labels = torch.ones((bs, 1), device=device)
            fake_labels = torch.zeros((bs, 1), device=device)

            d_optimizer.zero_grad()
            d_real = discriminator(r_data)
            with torch.no_grad():
                g_probs = torch.sigmoid(generator(z))
            d_fake = discriminator(g_probs)

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
            g_loss = -d_fake_for_g.mean()
            g_loss.backward()
            g_optimizer.step()

            if i == len(train_loader) - 1 and (epoch + 1) % 50 == 0:
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
                # imshow(torchvision.utils.make_grid(gen_data.cpu()[0]))


if __name__ == "__main__":
    main()
