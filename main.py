import numpy as np
import pennylane as qml
import tensorflow as tf

import warnings
from pennylane.exceptions import PennyLaneDeprecationWarning

warnings.simplefilter("ignore", PennyLaneDeprecationWarning)

device = qml.device("default.qubit", wires = 3)

def real(angles: tuple[float]) -> None:
    qml.Hadamard(wires = 0)
    qml.Rot(*angles, wires = 0)

def generator(w: tuple[float]) -> None:
    qml.Hadamard(wires = 0)
    qml.RX(w[0], wires = 0)
    qml.RX(w[1], wires = 1)
    qml.RY(w[2], wires = 0)
    qml.RY(w[3], wires = 1)
    qml.RZ(w[4], wires = 0)
    qml.RZ(w[5], wires = 1)
    qml.CNOT(wires = [0, 1])
    qml.RX(w[6], wires = 0)
    qml.RY(w[7], wires = 0)
    qml.RZ(w[8], wires = 0)

def discriminator(w: tuple[float]) -> None:
    qml.Hadamard(wires = 0)
    qml.RX(w[0], wires = 0)
    qml.RX(w[1], wires = 2)
    qml.RY(w[2], wires = 0)
    qml.RY(w[3], wires = 2)
    qml.RZ(w[4], wires = 0)
    qml.RZ(w[5], wires = 2)
    qml.CNOT(wires = [0, 2])
    qml.RX(w[6], wires = 2)
    qml.RY(w[7], wires = 2)
    qml.RZ(w[8], wires = 2)

@qml.qnode(device)
def real_disc_circuit(phi: float, theta: float, omega: float, disc_weights: tuple[float]) -> float:
    real([phi, theta, omega])
    discriminator(disc_weights)
    return qml.expval(qml.PauliZ(2))

@qml.qnode(device)
def gen_disc_circuit(gen_weights: tuple[float], disc_weights: tuple[float]) -> float:
    generator(gen_weights)
    discriminator(disc_weights)
    return qml.expval(qml.PauliZ(2))

def prob_real_true(disc_weights: tuple[float]) -> float:
    true_disc_output = real_disc_circuit(phi, theta, omega, disc_weights)
    # convert to probability
    prob_real_true = (true_disc_output + 1) / 2
    return prob_real_true


def prob_fake_true(gen_weights: tuple[float], disc_weights: tuple[float]) -> float:
    fake_disc_output = gen_disc_circuit(gen_weights, disc_weights)
    # convert to probability
    prob_fake_true = (fake_disc_output + 1) / 2
    return prob_fake_true


def disc_cost(gen_weights: tuple[float], disc_weights: tuple[float]) -> float:
    cost = prob_fake_true(gen_weights, disc_weights) - prob_real_true(disc_weights)
    return cost


def gen_cost(gen_weights: tuple[float], disc_weights: tuple[float]) -> float:
    return -prob_fake_true(gen_weights, disc_weights)

@qml.qnode(device)
def bloch_vector_real(angles: tuple[float]) -> list[float]:
    real(angles)
    return [qml.expval(o) for o in observable]

@qml.qnode(device)
def bloch_vector_generator(angles: tuple[float]) -> list[float]:
    generator(angles)
    return [qml.expval(o) for o in observable]

if __name__ == "__main__":
    # Define angles
    phi = np.pi / 6
    theta = np.pi / 2
    omega = np.pi / 7
    
    # Set seed, epsilon and steps
    np.random.seed(0)
    epsilon = 0.01
    steps = 20
    
    # Set weights
    init_gen_weights = np.array([np.pi] + [0] * 8) + \
                       np.random.normal(scale = epsilon, size = 9)
    init_disc_weights = np.random.normal(size = 9)
    
    gen_weights = tf.Variable(init_gen_weights)
    disc_weights = tf.Variable(init_disc_weights)
    
    # Create optimiser
    optimiser = tf.keras.optimizers.SGD(0.4)
    optimiser.build([disc_weights, gen_weights])
    
    # Optimise solely the discriminator
    cost = lambda: disc_cost(gen_weights, disc_weights)

    for step in range(steps):
        with tf.GradientTape() as tape:
            loss_value = cost()
        gradients = tape.gradient(loss_value, [disc_weights])
        optimiser.apply_gradients(zip(gradients, [disc_weights]))
        if step % 5 == 0:
            cost_val = loss_value.numpy()
            print(f"Step {step}: cost = {cost_val}")
    
    print("Discriminator classifies real data correctly:")
    print(f"P(real classified as real) = {prob_real_true(disc_weights).numpy()}")
    
    print("Generator classifies fake data:")
    print(f"P(fake classified as real) = {prob_fake_true(gen_weights, disc_weights).numpy()}")
    
    # Optimise the generator to fool discriminator
    cost = lambda: gen_cost(gen_weights, disc_weights)

    for step in range(steps):
        with tf.GradientTape() as tape:
            loss_value = cost()
        gradients = tape.gradient(loss_value, [gen_weights])
        optimiser.apply_gradients(zip(gradients, [gen_weights]))
        if step % 5 == 0:
            cost_val = loss_value.numpy()
            print(f"Step {step}: cost = {cost_val}")
    
    print("Discriminator classifies real data incorrectly:")
    print(f"P(fake classified as real) = {prob_fake_true(gen_weights, disc_weights).numpy()}")
    
    # When cost is close to zero, the discriminator assigns
    # the same probability to real and fake data
    print(f"Discriminator cost = {disc_cost(gen_weights, disc_weights).numpy()}")
    
    observable = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
    
    print(f"Real Bloch vector: {bloch_vector_real([phi, theta, omega])}")
    print(f"Generator Bloch vector: {bloch_vector_generator(gen_weights)}")
