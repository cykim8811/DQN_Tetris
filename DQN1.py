import PyTetris
import tensorflow.keras as keras
from tools import *
import numpy as np
from dlmodel import *

batch_size = 100

behavior_policy = policy_monte_e_greedy
target_policy = policy_monte_greedy

display = True
new_model = False  # min loss backup update
model_name = "Q8"
# Q5: after error fix, erase well, but too brave to survive
# Q7 548 epochs of training, batch200

if display:
    window = PyTetris.Window()
    window.set_ghost(0)
    window.set_gravity(0)
else:
    window = None

if new_model:
    Q = build_model()
    Q.compile(loss="MSE", optimizer=keras.optimizers.Adam(learning_rate=0.01))
else:
    Q = keras.models.load_model(model_name + ".h5")

S = PyTetris.State(10, 20)

X, Y = [], []

life_duration = 0
score_duration = 0
life_list = [1]
score_list = [0]

qvalue = Display()

epoch = 1
while True:
    for batch in range(batch_size):

        s0 = S.copy()
        s1, r1, Q1 = behavior_policy(S, Q)
        s2, r2, Q2 = target_policy(s1, Q)
        S = s1.copy()

        qvalue.print(
            ("[Epoch %d]\t" % epoch) + "#" * int(batch * 30 / batch_size) + "=" * (30 - int(batch * 30 / batch_size))
            + "\tQ:\t%.3f" % Q1)

        # qvalue.print("%s  ->  %s" % (qvalue.crop(Q1, 8), qvalue.crop(r1 + gamma * Q2, 8)))

        X.append(state_to_layer(s0, s1))
        Y.append(r1 + gamma * Q2)

        if r1 == -100:
            S = PyTetris.State(10, 20)
            life_list.append(life_duration)
            score_list.append(score_duration)
            life_duration = 0
            score_duration = 0
        else:
            life_duration += 1
            score_duration += r1

        if display:
            window.set_state(s0)
            if not window.tick(): break

    qvalue.clear()

    loss = Q.train_on_batch(np.asarray(X), np.asarray(Y))
    print("[Epoch %d]" % epoch, end='\t')
    print("loss:\t%.4f" % loss, end='\t')
    if len(life_list) > 5:
        life_list = life_list[-5:]
        score_list = score_list[-5:]
    print("E(life):\t%.4f" % (sum(life_list) / len(life_list)), end='\t')
    print("E(score):\t%.4f" % (sum(score_list) / len(score_list) - survive_bonus * sum(life_list) / len(life_list)))
    Q.save(model_name + ".h5")
    if epoch % 10 == 0:
        Q.save("./backup/" + model_name + "_" + str(int(epoch / 10)) + ".h5")
    X, Y = [], []
    epoch += 1
