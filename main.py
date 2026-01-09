#!/usr/bin/env python

import numpy as np
import pennylane as qml
import torch

# Option A (matches the demo): Cirq simulator device
# Requires: pip install pennylane-cirq cirq
# dev = qml.device("cirq.simulator", wires=3)

# Option B (simpler): PennyLane built-in simulator (no Cirq plugin needed)
dev = qml.device("default.qubit", wires=3)


def real(angles, **kwargs):
    qml.Hadamard(wires=0)
    qml.Rot(*angles, wires=0)


def generator(w, **kwargs):
    qml.Hadamard(wires=0)
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=1)
    qml.RY(w[2], wires=0)
    qml.RY(w[3], wires=1)
    qml.RZ(w[4], wires=0)
    qml.RZ(w[5], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(w[6], wires=0)
    qml.RY(w[7], wires=0)
    qml.RZ(w[8], wires=0)


def discriminator(w):
    qml.Hadamard(wires=0)
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=2)
    qml.RY(w[2], wires=0)
    qml.RY(w[3], wires=2)
    qml.RZ(w[4], wires=0)
    qml.RZ(w[5], wires=2)
    qml.CNOT(wires=[0, 2])
    qml.RX(w[6], wires=2)
    qml.RY(w[7], wires=2)
    qml.RZ(w[8], wires=2)


@qml.qnode(dev, interface="torch")
def real_disc_circuit(phi, theta, omega, disc_weights):
    real([phi, theta, omega])
    discriminator(disc_weights)
    return qml.expval(qml.PauliZ(2))


@qml.qnode(dev, interface="torch")
def gen_disc_circuit(gen_weights, disc_weights):
    generator(gen_weights)
    discriminator(disc_weights)
    return qml.expval(qml.PauliZ(2))


# fixed "real data" angles
phi = np.pi / 6
theta = np.pi / 2
omega = np.pi / 7


def prob_real_true(disc_weights):
    true_out = real_disc_circuit(phi, theta, omega, disc_weights)
    return (true_out + 1.0) / 2.0


def prob_fake_true(gen_weights, disc_weights):
    fake_out = gen_disc_circuit(gen_weights, disc_weights)
    return (fake_out + 1.0) / 2.0


def disc_cost(gen_weights, disc_weights):
    return prob_fake_true(gen_weights, disc_weights) - prob_real_true(disc_weights)


def gen_cost(gen_weights, disc_weights):
    return -prob_fake_true(gen_weights, disc_weights)


torch.manual_seed(0)
eps = 1e-2

init_gen = np.array([np.pi] + [0] * 8) + np.random.normal(scale=eps, size=(9,))
init_disc = np.random.normal(size=(9,))

gen_weights = torch.nn.Parameter(torch.tensor(init_gen, dtype=torch.float64))
disc_weights = torch.nn.Parameter(torch.tensor(init_disc, dtype=torch.float64))

# Stage 1: train discriminator, freeze generator
optD = torch.optim.SGD([disc_weights], lr=0.4)

for step in range(50):
    optD.zero_grad()
    loss = disc_cost(gen_weights.detach(), disc_weights)  # detach gen
    loss.backward()
    optD.step()

    if step % 5 == 0:
        print(f"Step {step}: cost = {loss.item()}")

print("Prob(real classified as real): ", prob_real_true(disc_weights).item())
print(
    "Prob(fake classified as real): ", prob_fake_true(gen_weights, disc_weights).item()
)

# Stage 2: train generator, freeze discriminator
optG = torch.optim.SGD([gen_weights], lr=0.4)

for step in range(50):
    optG.zero_grad()
    loss = gen_cost(gen_weights, disc_weights.detach())  # detach disc
    loss.backward()
    optG.step()

    if step % 5 == 0:
        print(f"Step {step}: cost = {loss.item()}")

print(
    "Prob(fake classified as real): ", prob_fake_true(gen_weights, disc_weights).item()
)
print("Discriminator cost: ", disc_cost(gen_weights, disc_weights).item())


obs = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]


@qml.qnode(dev, interface="torch")
def bloch_vector_real(angles):
    real(angles)
    return [qml.expval(o) for o in obs]


@qml.qnode(dev, interface="torch")
def bloch_vector_generator(angles):
    generator(angles)
    return [qml.expval(o) for o in obs]


print("Real Bloch vector: ", [v.item() for v in bloch_vector_real([phi, theta, omega])])
print(
    "Generator Bloch vector: ", [v.item() for v in bloch_vector_generator(gen_weights)]
)
