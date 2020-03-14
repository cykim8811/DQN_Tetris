from PyTetris import State, Window
import tensorflow.keras as keras
from dlmodel import build_model
from tools import *
from random import random
from math import *

model_name = "Q2"
"""Initialize Window"""
window = Window()
window.set_gravity(0)
window.set_ghost(0)

"""Initialize DNN model"""
try:
    # Q = keras.models.load_model("../models/Q21.h5")
    Q = keras.models.load_model("%s.h5" % model_name)
except:
    Q = build_model()
    Q.compile(loss="MSE", optimizer=keras.optimizers.Adam(learning_rate=0.02))

"""Initial States to train"""
N = 10
state_list = [State(10, 20) for _ in range(N)]

"""Parameters"""
alpha = 0.2
gamma = 0.9

gameover_penalty = -1000

"""
            Main Loop
"""
Q1, Q2 = DL_Model(Q), DL_Model(Q)
logger = Display()
while True:
    X, Y = [], []  # data to train
    A1 = []
    A1_transitions = []
    Q_list = []
    for episode in range(N):
        S0 = state_list[episode]
        transitions = S0.transitions()
        if not S0.hold_used:
            transitions.append((S0.holded(), (-1, -1, -1), 0))
        A1_transitions.append(transitions)
        Q1.put([state_to_layer(state_list[episode], x[0]) for x in transitions])

    Q1.evaluate()

    for episode in range(N):
        Q_values = Q1.get().flatten()
        e = 0.05 + 0.6 * episode / N
        ind = np.argmax(Q_values) if random() < (1 - e) else floor(random() * len(Q_values))
        Q_list.append(Q_values[ind])
        A1.append(A1_transitions[episode][ind])  # Behavior policy : e-Greedy
        S1 = A1[episode][0]
        transitions = S1.transitions()
        if not S1.hold_used:
            transitions.append((S1.holded(), (-1, -1, -1), 0))
        Q2.put([state_to_layer(state_list[episode], x[0]) for x in transitions])

    Q1.clear()
    Q2.evaluate()

    for episode in range(N):
        Q_values = Q2.get().flatten()
        Q_max = np.max(Q_values)  # Target policy : Greedy
        if A1[episode][1] != (-1, -1, -1) and A1[episode][1][1] < 0:
            r = gameover_penalty
        else:
            r = A1[episode][2]
        X.append(state_to_layer(state_list[episode], A1[episode][0]))
        Y.append(r + gamma * Q_max)
        if A1[episode][1] != (-1, -1, -1) and A1[episode][1][1] < 0:
            state_list[episode] = State(10, 20)
        else:
            state_list[episode] = A1[episode][0]

    Q2.clear()

    # train model
    loss = Q.train_on_batch(np.asarray(X), np.asarray(Y))
    logger.print("loss:\t%.4f\tQ:\t%.4f" % (loss, Q_list[0]))

    window.set_state(state_list[0])
    if not window.tick(): break
    Q.save("%s.h5" % model_name)

logger.clear()
