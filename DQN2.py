from PyTetris import State
import tensorflow.keras as keras
from tools import *
import numpy as np
from dlmodel import *
from time import time, sleep
import random

display = True
new_model = True
model_name = "Q15"
history = -1
# Q14 well working

batch_size = 20  # train multiple times on one dataset

branch = [5, 15]

gameover_penalty = -100
survive_bonus = 0.1


def tick(target: PyTetris.Window):
    while target.tick():
        sleep(0.01)


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
S = State(10, 20)

X, Y = [], []

if history > 0:
    Q = keras.models.load_model("./backup/" + model_name + "_%d.h5" % history)

life_duration = 0
score_duration = 0
life_list = []
score_list = []

qvalue = Display()

epoch = 1
while True:
    for batch in range(batch_size):
        t0 = time()
        transition_sample = policy_greedy_e_sample(S, Q, branch[0])
        nodes = [[[x[0], None, x[2] if (x[1][1] >= 0 or x[1] == (-1, -1, -1)) else gameover_penalty,
                   []] for x in transition_sample]]

        for depth in range(len(branch) - 1):
            next_node = []
            # paralleled policy_greedy_samples
            assign = []
            input_list = []
            transition_list = []
            for i, s in enumerate(nodes[-1]):
                transitions = s[0].transitions()
                transition_list.append(transitions)
                if not s[0].hold_used: transitions.append((s[0].holded(), (-1, -1, -1), 0))
                inputs = [state_to_layer(s[0], s1[0]) for s1 in transitions]
                input_list += inputs
                assign.append(len(inputs))

            Q_values = Q.predict(np.asarray(input_list))

            transition_ind = 0
            for i, s in enumerate(nodes[-1]):
                transitions_sorted = next(
                    zip(*sorted(list(zip(transition_list[i], Q_values[transition_ind:transition_ind + assign[i]])),
                                key=lambda x: x[1], reverse=True)))
                transition_ind += assign[i]
                transition_sample = transitions_sorted[:min(len(transitions_sorted), branch[depth + 1])]
                # /paralleled policy_greedy_samples
                for t in transition_sample:
                    r = t[2] if (t[1][1] >= 0 or t[1] == (-1, -1, -1)) else gameover_penalty
                    next_node.append([t[0], i, r, []])
            nodes.append(next_node)

        # calculate Q
        assign = []
        predict_input = []
        for leaf in nodes[-1]:
            leaf_transitions = leaf[0].transitions()
            assign.append(len(leaf_transitions))
            predict_input += [state_to_layer(leaf[0], x[0]) for x in leaf_transitions]

        predict_output = Q.predict(np.asarray(predict_input)).flatten()

        assign_ind = 0
        for i, leaf in enumerate(nodes[-1]):
            Q_values = predict_output[assign_ind: assign_ind + assign[i]]
            assign_ind += assign[i]
            Q_average = max(Q_values) * 0.8 + ((sum(Q_values) - max(Q_values)) / (len(Q_values) - 1)) * 0.2
            nodes[-2][leaf[1]][3].append(leaf[2] + gamma * Q_average)

        final_Q = []

        for i in reversed(range(len(nodes) - 1)):
            for node in nodes[i]:
                Q_average = max(node[3]) * 0.8 + ((sum(node[3]) - max(node[3])) / (len(node[3]) - 1)) * 0.2
                prop = node[2] + gamma * Q_average
                if i > 0:
                    nodes[i - 1][node[1]][3].append(prop)
                else:
                    final_Q.append(prop)

        for i, node in enumerate(nodes[0]):
            X.append(state_to_layer(S, node[0]))
            Y.append(final_Q[i])

        a = np.argmax(np.asarray(final_Q))
        if random.random() < 0.95:  # e-greedy
            S = nodes[0][a][0].copy()
        else:
            S = random.choice(nodes[0])[0].copy()

        r = nodes[0][a][2]

        qvalue.print(
            ("[Epoch %d]\t" % epoch) + "#" * int(batch * 30 / batch_size) + "=" * (30 - int(batch * 30 / batch_size))
            + "\tTime/Step:\t%.3f" % (time() - t0)
            + "\tQ:\t%.3f" % max(final_Q)
            + "\tr:\t%.2f" % (score_duration)
            + "\tlife:\t%d" % (life_duration)
        )

        if r == -100:
            S = PyTetris.State(10, 20)
            life_list.append(life_duration)
            score_list.append(score_duration)
            life_duration = 0
            score_duration = 0
        else:
            life_duration += 1
            score_duration += r

        if display:
            window.set_state(S)
            if not window.tick(): break

    qvalue.clear()

    loss = Q.train_on_batch(np.asarray(X), np.asarray(Y))
    print("[Epoch %d]" % epoch, end='\t')
    print("loss:\t%.4f" % loss, end='\t')
    if len(life_list) > 5:
        life_list = life_list[-5:]
        score_list = score_list[-5:]
    print("E(life):\t%.4f" % (sum(life_list) / len(life_list) if len(life_list) > 0 else 0), end='\t')
    print("E(score):\t%.4f" % (sum(score_list) / len(score_list) if len(score_list) > 0 else 0))
    Q.save(model_name + ".h5")
    if epoch % 10 == 0:
        Q.save("./backup/" + model_name + "_" + str(int(epoch / 10)) + ".h5")
    X = X[int(len(X) * 0.75):]
    Y = Y[int(len(Y) * 0.75):]
    epoch += 1
