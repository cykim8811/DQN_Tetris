import PyTetris
import tensorflow.keras as keras
import random
import numpy as np
from math import *

alpha = 0.05
gamma = 0.8

epsilon = 0.1


def policy_greedy_e_sample(S: PyTetris.State, model: keras.Model, n: int):
    e = 0.4
    transitions = S.transitions()
    d_greedy = min(floor(n * (1 - e)), len(transitions))
    d_random = min(ceil(n * e), len(transitions))
    if not S.hold_used: transitions.append((S.holded(), (-1, -1, -1), 0))
    inputs = np.asarray([state_to_layer(S, s1[0]) for s1 in transitions])
    Q_values = model.predict(inputs)
    transitions_sorted = next(zip(*sorted(list(zip(transitions, Q_values)), key=lambda x: x[1], reverse=True)))
    return list(transitions_sorted[:d_greedy]) + list(random.sample(transitions_sorted[d_greedy:], d_random))


def policy_greedy_sample(S: PyTetris.State, model: keras.Model, n: int):
    transitions = S.transitions()
    if not S.hold_used: transitions.append((S.holded(), (-1, -1, -1), 0))
    inputs = np.asarray([state_to_layer(S, s1[0]) for s1 in transitions])
    Q_values = model.predict(inputs)
    transitions_sorted = next(zip(*sorted(list(zip(transitions, Q_values)), key=lambda x: x[1], reverse=True)))
    return transitions_sorted[:n]


def policy_random_sample(S: PyTetris.State, model: keras.Model, n: int):
    transitions = S.transitions()
    if not S.hold_used: transitions.append((S.holded(), (-1, -1, -1), 0))
    return random.sample(transitions, min(n, len(transitions)))


def state_to_layer(s0: PyTetris.State, s1: PyTetris.State):
    layer = np.zeros((10, 20, 33), dtype=np.float16)
    layer[:, :, 0] = (s0.get_screen() == 0)
    layer[:, :, 1] = (s1.get_screen() == 0)
    for i in range(3):
        for t in range(7):
            layer[:, :, 2 + i * 7 + t] = np.ones((10, 20)) * (s1.get_block_next(i) == t)

    for t in range(7):
        layer[:, :, 2 + 3 * 7 + t] = np.ones((10, 20)) * (s1.block_hold == t)

    layer[:, :, 30] = np.ones((10, 20)) * s1.hold_used
    layer[:, :, 31] = np.ones((10, 20)) * s1.combo
    layer[:, :, 32] = np.ones((10, 20)) * s1.btb

    return layer


class Display:
    def __init__(self):
        self.l = 0

    def print(self, str):
        print('\b' * self.l, end='')
        print(str, end='')
        self.l = len(str)

    def clear(self):
        print('\b' * self.l, end='')

    def crop(self, f, l):
        return str(f)[:l] + '0' * ((l - len(str(f))) * (len(str(f)) > l))
