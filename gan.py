#!/usr/bin/env python

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


SIDE_LENGTH = 3
ITERATIONS = 10000
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
            nn.Linear(16, 1),
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


def eval_discriminator(
    generator, discriminator, real_loader, device, n_fake=512, thresh=0.5
):
    discriminator.eval()
    generator.eval()

    # --- real stats ---
    real_scores = []
    with torch.no_grad():
        for real, _ in real_loader:
            real = real.to(device)
            real_scores.append(discriminator(real).view(-1))
    real_scores = torch.cat(real_scores, dim=0)

    p_real_given_real = (real_scores > thresh).float().mean().item()

    # --- fake stats ---
    fake_scores = []
    fake_pixels = []
    with torch.no_grad():
        bs = 64
        for _ in range((n_fake + bs - 1) // bs):
            z = torch.randn(bs, GENERATOR_INPUT_SIZE, device=device)
            fake = generator(z)
            fake_pixels.append(fake.reshape(-1))
            fake_scores.append(discriminator(fake).view(-1))
    fake_scores = torch.cat(fake_scores, dim=0)[:n_fake]
    fake_pixels = torch.cat(fake_pixels, dim=0)

    p_real_given_fake = (fake_scores > thresh).float().mean().item()
    p_fake_given_fake = 1.0 - p_real_given_fake

    out = {
        "p_real_given_real": p_real_given_real,
        "p_real_given_fake": p_real_given_fake,
        "p_fake_given_fake": p_fake_given_fake,
        "real_scores": real_scores.detach().cpu(),
        "fake_scores": fake_scores.detach().cpu(),
        "fake_pixels": fake_pixels.detach().cpu(),
    }

    discriminator.train()
    generator.train()
    return out


def is_bars_or_stripes(x_hard):
    """
    x_hard: (B, 3, 3) in {0,1}
    bars: each column constant across rows
    stripes: each row constant across cols
    """
    # bars: all rows equal for each col -> each col has zero variance over rows
    bars = (x_hard.var(dim=1) == 0).all(dim=1)  # (B,)
    # stripes: all cols equal for each row -> each row has zero variance over cols
    stripes = (x_hard.var(dim=2) == 0).all(dim=1)  # (B,)
    return bars | stripes


def eval_generator_validity(generator, device, n=1024):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n, GENERATOR_INPUT_SIZE, device=device)
        probs = generator(z)  # (n,3,3) in [0,1]
        hard = (probs > 0.5).float()
        valid = is_bars_or_stripes(hard).float().mean().item()
    generator.train()
    return valid


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
    p_rr_hist = []  # P(real|real)
    p_rf_hist = []  # P(real|fake)
    valid_hist = []  # fraction of valid bars/stripes among hard samples
    steps_hist = []  # x-axis

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
                stats = eval_discriminator(
                    generator, discriminator, train_loader, device, n_fake=512
                )
                valid = eval_generator_validity(generator, device, n=1024)

                p_rr_hist.append(stats["p_real_given_real"])
                p_rf_hist.append(stats["p_real_given_fake"])
                valid_hist.append(valid)
                steps_hist.append(epoch + 1)

                print(
                    f"Epoch {epoch + 1} | "
                    f"Loss_D: {d_loss.item():.4f} | Loss_G: {g_loss.item():.4f} | "
                    f"P(real|real): {stats['p_real_given_real']:.3f} | "
                    f"P(real|fake): {stats['p_real_given_fake']:.3f} | "
                    f"valid@0.5: {valid:.3f}"
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

    # Final samples (hard thresholded)
    with torch.no_grad():
        fig, axs = plt.subplots(4, 4)
        for row in axs:
            for ax in row:
                sample = generator(torch.randn(1, GENERATOR_INPUT_SIZE, device=device))[
                    0
                ]
                hard = (sample > 0.5).float()
                ax.imshow(sample.detach().cpu().numpy(), cmap="gray")
        plt.show()

    # Loss curves
    plt.figure()
    avg = 50
    if len(d_losses) > avg:
        plt.plot(
            np.linspace(0, len(g_losses), len(d_losses) - avg + 1),
            np.convolve(d_losses, np.ones(avg) / avg, mode="valid"),
            label="critic",
        )
    if len(g_losses) > avg:
        plt.plot(
            np.convolve(g_losses, np.ones(avg) / avg, mode="valid"), label="generator"
        )
    plt.legend()

    # --- Accuracy-style curves ---
    plt.figure()
    plt.plot(steps_hist, p_rr_hist, label="P(real|real)")
    plt.plot(steps_hist, p_rf_hist, label="P(real|fake) (G fool rate)")
    plt.plot(steps_hist, valid_hist, label="G valid bars/stripes (hard@0.2)")
    plt.ylim(0, 1.05)
    plt.legend()

    # --- Discriminator score distributions + pixel probability distribution ---
    final_stats = eval_discriminator(
        generator, discriminator, train_loader, device, n_fake=2048
    )

    plt.figure()
    plt.hist(final_stats["fake_pixels"].numpy(), bins=30)
    plt.title("Generator pixel probability distribution")
    plt.show()


if __name__ == "__main__":
    main()
