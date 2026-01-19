#!/usr/bin/env python3
# QGAN for Bars & Stripes (3x3) — ADVERSARIAL (NO KL)
#
# Generator loss (adversarial + entropy + BAS-mass reward):
#   L_G = - E_{x~p}[ D(x) ]  - lam * H(p)  - gamma * chi
# where:
#   H(p)  = - sum_x p(x) log p(x)
#   chi   = sum_{x in BAS} p(x)
#
# Discriminator loss: hinge (spectral norm MLP)
#
# Requirements: pennylane, jax, optax, torch, numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm

import jax
import jax.numpy as jnp
import optax
import pennylane as qml

from pathlib import Path

jax.config.update("jax_enable_x64", True)

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


# ----------------------------
# Dataset: Bars & Stripes NxN
# ----------------------------
def get_bars_and_stripes(n: int) -> np.ndarray:
    bitstrings = [list(np.binary_repr(i, n))[::-1] for i in range(2**n)]
    bitstrings = np.array(bitstrings, dtype=int)

    stripes = bitstrings.copy()
    stripes = np.repeat(stripes, n, 0)
    stripes = stripes.reshape(2**n, n * n)

    bars = bitstrings.copy()
    bars = bars.reshape(2**n * n, 1)
    bars = np.repeat(bars, n, 1)
    bars = bars.reshape(2**n, n * n)

    return np.vstack((stripes[0 : stripes.shape[0] - 1], bars[1 : bars.shape[0]]))


# ----------------------------
# Discriminator (PyTorch) — logits (NO sigmoid)
# ----------------------------
class Discriminator(nn.Module):
    def __init__(self, dim=9):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(9, 32)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(32, 1)),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Helpers: basis enumeration + indexing (qml.probs ordering)
# ----------------------------
def all_bitstrings_big_endian(dim: int) -> np.ndarray:
    # index i corresponds to binary_repr(i,width=dim) = [MSB..LSB]
    return np.array(
        [list(map(int, np.binary_repr(i, width=dim))) for i in range(2**dim)],
        dtype=np.float32,
    )


def bits_to_index_big_endian(bits01: np.ndarray) -> int:
    bits01 = np.asarray(bits01, dtype=int).ravel()
    pows = 2 ** np.arange(len(bits01) - 1, -1, -1)
    return int(np.dot(bits01, pows))


# ----------------------------
# QGAN (NO KL)
# ----------------------------
class QGAN:
    def __init__(
        self,
        n: int = 3,
        n_layers: int = 8,
        # learning rates
        g_lr: float = 0.008,
        d_lr: float = 1e-3,
        # update ratio
        d_steps: int = 5,
        g_steps: int = 1,
        batch_size: int = 128,
        # regularizers (annealed)
        entropy_lam_init: float = 0.05,
        mass_gamma_init: float = 0.0,
        entropy_lam_final: float = 0.005,
        mass_gamma_final: float = 0.8,
        # anneal windows
        anneal_start: int = 400,
        anneal_end: int = 900,
        # metrics
        coverage_thresh: float = 1e-3,
        seed: int = 0,
        device_torch: str = "cpu",
    ):
        self.n = n
        self.dim = n * n
        self.n_qubits = self.dim
        self.n_layers = n_layers

        self.d_steps = d_steps
        self.g_steps = g_steps
        self.batch_size = batch_size

        self.entropy_lam_init = entropy_lam_init
        self.mass_gamma_init = mass_gamma_init
        self.entropy_lam_final = entropy_lam_final
        self.mass_gamma_final = mass_gamma_final

        self.anneal_start = anneal_start
        self.anneal_end = anneal_end

        self.coverage_thresh = coverage_thresh
        self.device_torch = device_torch

        rng = np.random.default_rng(seed)

        # Real data (BAS)
        self.data = get_bars_and_stripes(n).astype(np.float32)
        self.num_real = len(self.data)

        # Indices of BAS states (matching qml.probs indexing)
        self.real_indices = np.array(
            [bits_to_index_big_endian(x) for x in self.data], dtype=np.int32
        )

        # BAS mask for chi term
        bas_mask = np.zeros(2**self.dim, dtype=np.float64)
        bas_mask[self.real_indices] = 1.0
        self.bas_mask = bas_mask

        # Discriminator
        self.D = Discriminator(dim=self.dim).to(device_torch)
        self.d_opt = optim.Adam(self.D.parameters(), lr=d_lr)

        # All basis states corresponding to qml.probs ordering
        self.all_x_np = all_bitstrings_big_endian(self.dim)
        self.all_x_torch = torch.tensor(
            self.all_x_np, dtype=torch.float32, device=device_torch
        )

        # Quantum generator probs qnode (JAX)
        self.dev_probs = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev_probs, interface="jax")
        def gen_probs(weights):
            qml.StronglyEntanglingLayers(
                weights=weights,
                ranges=[1] * self.n_layers,
                wires=range(self.n_qubits),
            )
            return qml.probs(wires=range(self.n_qubits))

        self.gen_probs = jax.jit(gen_probs)

        # Generator params (mild init helps stability)
        wshape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.n_layers, n_wires=self.n_qubits
        )
        self.g_params = rng.random(size=wshape)

        # Generator optimizer
        self.g_opt = optax.adam(learning_rate=g_lr)
        self.g_opt_state = self.g_opt.init(self.g_params)

        # Sampling circuit (for discriminator batches)
        def _sample_circuit(weights):
            qml.StronglyEntanglingLayers(
                weights=weights,
                ranges=[1] * self.n_layers,
                wires=range(self.n_qubits),
            )
            return qml.sample(wires=range(self.n_qubits))

        self._sample_circuit = _sample_circuit

        # current annealed coeffs
        self.entropy_lam = entropy_lam_init
        self.mass_gamma = mass_gamma_init

        # cache mask as jax array
        self.bas_mask_jax = jnp.array(self.bas_mask)

    # ----------------------------
    # Annealing schedule
    # ----------------------------
    def update_anneal(self, epoch: int):
        if epoch <= self.anneal_start:
            t = 0.0
        elif epoch >= self.anneal_end:
            t = 1.0
        else:
            t = (epoch - self.anneal_start) / (self.anneal_end - self.anneal_start)

        self.entropy_lam = (1 - t) * self.entropy_lam_init + t * self.entropy_lam_final
        self.mass_gamma = (1 - t) * self.mass_gamma_init + t * self.mass_gamma_final

    # ----------------------------
    # Sampling from generator
    # ----------------------------
    def sample_generator(self, shots: int) -> np.ndarray:
        dev_samp = qml.device("default.qubit", wires=self.n_qubits, shots=shots)
        qnode = qml.QNode(self._sample_circuit, device=dev_samp)
        return np.array(qnode(self.g_params), dtype=np.float32)

    # ----------------------------
    # Discriminator step (hinge)
    # ----------------------------
    @staticmethod
    def d_hinge_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.relu(1.0 - d_real)) + torch.mean(torch.relu(1.0 + d_fake))

    def d_step(self) -> float:
        self.D.train()

        idx = np.random.randint(0, self.num_real, size=self.batch_size)
        real = torch.tensor(self.data[idx], dtype=torch.float32, device=self.device_torch)

        fake_np = self.sample_generator(shots=self.batch_size)
        fake = torch.tensor(fake_np, dtype=torch.float32, device=self.device_torch)

        self.d_opt.zero_grad()
        d_real = self.D(real)
        d_fake = self.D(fake)
        gp = gradient_penalty(self.D, real, fake, device=self.device_torch)
        loss = (d_fake.mean() - d_real.mean()) + 5.0 * gp
        loss.backward()
        self.d_opt.step()

        return float(loss.item())

    # ----------------------------
    # Generator step (JAX)
    # L_G = -E_p[D(x)] - lam*H(p) - gamma*chi
    # ----------------------------
    @staticmethod
    def entropy(p: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
        p = jnp.clip(p, eps, 1.0)
        return -jnp.sum(p * jnp.log(p))

    def g_step(self) -> float:
        self.D.eval()
        with torch.no_grad():
            Dx_logits_np = (
                self.D(self.all_x_torch).squeeze(1).cpu().numpy().astype(np.float64)
            )
        Dx = jnp.array(Dx_logits_np)

        lam = float(self.entropy_lam)
        gamma = float(self.mass_gamma)
        bas_mask = self.bas_mask_jax

        def loss_fn(params):
            p = self.gen_probs(params)  # (512,)
            adv = -jnp.sum(p * Dx)
            ent = self.entropy(p)
            chi = jnp.sum(p * bas_mask)
            return adv - lam * ent - gamma * chi

        loss_val, grads = jax.value_and_grad(loss_fn)(self.g_params)
        updates, self.g_opt_state = self.g_opt.update(grads, self.g_opt_state)
        self.g_params = optax.apply_updates(self.g_params, updates)
        return float(loss_val)

    # ----------------------------
    # Metrics
    # ----------------------------
    def probs(self) -> np.ndarray:
        return np.array(self.gen_probs(self.g_params))

    def chi_prob_mass(self) -> float:
        p = self.probs()
        return float(p[self.real_indices].sum())

    def coverage(self) -> float:
        p = self.probs()
        bas_p = p[self.real_indices]
        return float(np.mean(bas_p > self.coverage_thresh))

    def neff_bas(self) -> float:
        p = self.probs()
        bas_p = p[self.real_indices]
        s = bas_p.sum()
        if s <= 1e-12:
            return 0.0
        q = bas_p / s
        return float(1.0 / np.sum(q**2))

    # ----------------------------
    # Train loop
    # ----------------------------
    def train(self, epochs: int = 1200, print_every: int = 20, early_stop: bool = True):
        hist = {"d_loss": [], "g_loss": [], "chi": [], "coverage": [], "neff": []}

        best_score = -1e9
        best_params = None
        patience = 120
        bad = 0

        steps_per_epoch = max(1, self.num_real // self.batch_size)

        for ep in range(1, epochs + 1):
            self.update_anneal(ep)

            for _ in range(steps_per_epoch):
                for _ in range(self.d_steps):
                    d_loss = self.d_step()
                for _ in range(self.g_steps):
                    g_loss = self.g_step()

            chi = self.chi_prob_mass()
            cov = self.coverage()
            neff = self.neff_bas()

            hist["d_loss"].append(d_loss)
            hist["g_loss"].append(g_loss)
            hist["chi"].append(chi)
            hist["coverage"].append(cov)
            hist["neff"].append(neff)

            # Score: prioritize chi, keep diversity (neff), keep coverage
            score = (chi * 2.5) + (neff / 14.0) + (0.3 * cov)

            if score > best_score:
                best_score = score
                best_params = np.array(self.g_params, copy=True)
                bad = 0
            else:
                bad += 1

            if ep % print_every == 0 or ep == 1:
                print(
                    f"epoch {ep:4d} | "
                    f"D_loss {d_loss:.4f} | G_loss {g_loss:.4f} | "
                    f"chi {chi:.4f} | coverage {cov:.2f} | N_eff {neff:.2f} | "
                    f"lam {self.entropy_lam:.4f} | gamma {self.mass_gamma:.3f}"
                )

            if early_stop and ep > 250 and bad >= patience:
                print(
                    f"Early stop at epoch {ep} (no improvement for {patience} epochs). "
                    f"Restoring best generator."
                )
                self.g_params = best_params
                break

        return hist

    # ----------------------------
    # Visualization
    # ----------------------------
    def plot_samples(self, shots: int = 64, title: str = "Generator samples", savepath: Path | None = None):
        samples = self.sample_generator(shots=shots)
        mask = np.any(np.all(samples[:, None] == self.data[None, :, :], axis=2), axis=1)

        plt.figure(figsize=(8, 8))
        j = 1
        for img, ok in zip(samples[:64], mask[:64]):
            ax = plt.subplot(8, 8, j)
            j += 1
            plt.imshow(img.reshape(self.n, self.n), cmap="gray", vmin=0, vmax=1)
            if not ok:
                plt.setp(ax.spines.values(), color="red", linewidth=1.5)
            plt.xticks([])
            plt.yticks([])

        plt.suptitle(title)
        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, dpi=300, bbox_inches="tight")

        plt.show()
        print(f"sampled chi (shots={shots}): {float(mask.mean()):.4f}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    n = 3

    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    qgan = QGAN(
        n=n,
        n_layers=8,
        g_lr=0.008,
        d_lr=1e-3,
        d_steps=5,
        g_steps=1,
        batch_size=128,
        entropy_lam_init=0.05,
        mass_gamma_init=0.0,
        entropy_lam_final=0.005,
        mass_gamma_final=0.8,
        anneal_start=400,
        anneal_end=900,
        coverage_thresh=1e-3,
        seed=0,
        device_torch="cpu",
    )

    # Target support visualization (optional): BAS indices only
    # (No KL target distribution used; this plot is just illustrative of BAS states count.)
    plt.figure(figsize=(12, 4))
    pi_like = np.zeros(2 ** (n * n))
    pi_like[qgan.real_indices] = 1.0 / len(qgan.real_indices)
    plt.bar(np.arange(2 ** (n * n)), pi_like, width=1.0)
    plt.title("BAS support (uniform over BAS indices) — for reference")
    plt.xlabel("Basis state index (qml.probs ordering)")
    plt.ylabel("Probability (reference)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "bas_support_reference.png", dpi=300)
    plt.show()

    before_metrics = {
        "chi": qgan.chi_prob_mass(),
        "coverage": qgan.coverage(),
        "N_eff": qgan.neff_bas(),
    }

    print("Before training:")
    for k, v in before_metrics.items():
        print(f"  {k}: {v}")

    qgan.plot_samples(
        shots=64,
        title="Before training",
        savepath=RESULTS_DIR / "samples_before.png",
    )

    hist = qgan.train(epochs=1500, print_every=20, early_stop=True)

    after_metrics = {
        "chi": qgan.chi_prob_mass(),
        "coverage": qgan.coverage(),
        "N_eff": qgan.neff_bas(),
    }

    print("After training:")
    for k, v in after_metrics.items():
        print(f"  {k}: {v}")

    qgan.plot_samples(
        shots=64,
        title="After training",
        savepath=RESULTS_DIR / "samples_after.png",
    )

    with open(RESULTS_DIR / "metrics.txt", "w") as f:
        f.write("=== BEFORE TRAINING ===\n")
        for k, v in before_metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== AFTER TRAINING ===\n")
        for k, v in after_metrics.items():
            f.write(f"{k}: {v}\n")

    # Curves
    plt.figure()
    plt.plot(hist["chi"])
    plt.title("chi (mass on BAS)")
    plt.xlabel("Epoch")
    plt.ylabel("chi")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "chi_curve.png", dpi=300)
    plt.show()

    plt.figure()
    plt.plot(hist["coverage"])
    plt.title("coverage (BAS modes with p > thresh)")
    plt.xlabel("Epoch")
    plt.ylabel("coverage")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "coverage_curve.png", dpi=300)
    plt.show()

    plt.figure()
    plt.plot(hist["neff"])
    plt.title("N_eff (effective BAS modes)")
    plt.xlabel("Epoch")
    plt.ylabel("N_eff")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "neff_curve.png", dpi=300)
    plt.show()

    plt.figure()
    plt.plot(hist["d_loss"])
    plt.title("Discriminator hinge loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "d_loss_curve.png", dpi=300)
    plt.show()

    plt.figure()
    plt.plot(hist["g_loss"])
    plt.title("Generator loss (no KL)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "g_loss_curve.png", dpi=300)
    plt.show()
