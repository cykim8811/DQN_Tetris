import PyTetris
import tensorflow.keras as keras
import random
import numpy as np

gameover_penalty = 100
survive_bonus = 0.1

alpha = 0.05
gamma = 0.8

epsilon = 0.1

branch_n = 8
search_depth = 2


def state_to_layer(s0: PyTetris.State, s1: PyTetris.State):
    layer = np.zeros((10, 20, 33))
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


def policy_greedy(S: PyTetris.State, model: keras.Model):
    global alpha, gamma
    transitions = S.transitions()
    if not S.hold_used: transitions.append((S.holded(), (-1, -1, -1), 0))
    if len(transitions) == 0:
        print("Warning: len(transitions == 0")
        return PyTetris.State(10, 20), -gameover_penalty, -gameover_penalty / (1 - gamma)
    inputs = np.asarray([state_to_layer(S, st[0]) for st in transitions])
    values = model.predict(inputs).flatten()
    target = transitions[np.argmax(values)]
    if not target[1] == (-1, -1, -1) and target[1][1] < 0:
        return target[0], - gameover_penalty, max(values)
    else:
        return target[0], target[2] + survive_bonus, max(values)


def policy_random(S: PyTetris.State, model: keras.Model):
    transitions = S.transitions()
    if not S.hold_used: transitions.append((S.holded(), (-1, -1, -1), 0))
    if len(transitions) == 0:
        print("Warning: len(transitions == 0")
        return PyTetris.State(10, 20), -gameover_penalty, -gameover_penalty / (1 - gamma)
    target_ind = random.randint(0, len(transitions) - 1)
    inputs = np.asarray([state_to_layer(S, transitions[target_ind][0])])
    values = model.predict(inputs).flatten()[0]
    target = transitions[target_ind]
    if not target[1] == (-1, -1, -1) and target[1][1] < 0:
        return target[0], - gameover_penalty, values
    else:
        return target[0], target[2] + survive_bonus, values


def policy_e_greedy(S: PyTetris.State, model: keras.Model):
    global alpha, gamma, epsilon
    if random.random() < 1 - epsilon:
        return policy_greedy(S, model)
    else:
        return policy_random(S, model)


def monte_recursive(S: PyTetris.State, model: keras.Model, depth=2):
    if depth == 1:
        return policy_greedy(S, model)[2]
    else:
        transitions = S.transitions()
        if not S.hold_used: transitions.append((S.holded(), (-1, -1, -1), 0))
        inputs = np.asarray([state_to_layer(S, x[0]) for x in transitions])
        tr_val = list(zip(transitions, model.predict(inputs).flatten()))
        tr_val.sort(key=lambda x: -x[1])
        transitions = [x[0] for x in tr_val[:branch_n]]
        res = [monte_recursive(x[0], model, depth - 1) for x in transitions]
        return max(res) * 0.8 + 0.2 * (sum(res) - max(res)) / (len(res) - 1)


def policy_monte_greedy(S: PyTetris.State, model: keras.Model):
    global alpha, gamma
    transitions = S.transitions()
    if not S.hold_used: transitions.append((S.holded(), (-1, -1, -1), 0))
    if len(transitions) == 0:
        print("Warning: len(transitions == 0")
        return PyTetris.State(10, 20), -gameover_penalty, -gameover_penalty / (1 - gamma)
    inputs = np.asarray([state_to_layer(S, x[0]) for x in transitions])
    tr_val = list(zip(transitions, model.predict(inputs).flatten()))
    tr_val.sort(key=lambda x: -x[1])
    transitions = [x[0] for x in tr_val[:branch_n]]
    monte_values = np.asarray([monte_recursive(s[0], model, search_depth) for s in transitions])
    target_ind = np.argmax(monte_values)
    target = transitions[target_ind]
    if not target[1] == (-1, -1, -1) and target[1][1] < 0:
        return target[0], - gameover_penalty, tr_val[target_ind][1]
    else:
        return target[0], target[2] + survive_bonus, tr_val[target_ind][1]


def policy_monte_e_greedy(S: PyTetris.State, model: keras.Model):
    global alpha, gamma, epsilon
    if random.random() < 1 - epsilon:
        return policy_monte_greedy(S, model)
    else:
        return policy_random(S, model)


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
