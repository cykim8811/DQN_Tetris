
import PyTetris
import tensorflow.keras as keras
from tools import *
import numpy as np
from dlmodel import *

batch_size = 100

behavior_policy = policy_e_greedy
target_policy = policy_greedy

display = True
new_model = True
model_name = "Q4"

if display:
    window = PyTetris.Window()
    window.set_ghost(0)
    window.set_gravity(0)
else:
    window = None

if new_model:
    Q = build_model()
    Q.compile(loss="MSE", optimizer=keras.optimizers.Adam(learning_rate=0.02))
else:
    Q = keras.models.load_model(model_name + ".h5")

S = PyTetris.State(10, 20)

train_r = [None]
train_S = [S]
train_Q = [None]

X, Y = [],  []

epoch = 1
while True:
    for batch in range(batch_size):
        s0 = S.copy()
        s1, r1, Q1 = behavior_policy(S, Q)
        s2, r2, Q2 = target_policy(s1, Q)

        train_r.append(r1)
        train_Q.append(Q2)
        train_S.append(s1)

        S = s1.copy()

        if len(train_S) > 6:
            train_r = train_r[1:]
            train_Q = train_Q[1:]
            train_S = train_S[1:]

        for i in range(1, len(train_S)):
            dist = len(train_S) - i
            X.append(state_to_layer(train_S[i-1], train_S[i]))
            t = Q2
            for j in reversed(range(i, len(train_S))):
                t = t * gamma + train_r[j]
            Y.append(t)

        if r1 == -gameover_penalty:
            train_r = [None]
            train_S = [S]
            train_Q = [None]

        if display:
            window.set_state(s0)
            if not window.tick(): break

    loss = Q.train_on_batch(np.asarray(X), np.asarray(Y))
    print("[Epoch %d]\tloss:\t%.4f" % (epoch, loss))
    Q.save(model_name + ".h5")
    X, Y = [], []
    epoch += 1
