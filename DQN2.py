from PyTetris import State
import tensorflow.keras as keras
from tools import *
import numpy as np
from dlmodel import *
from time import time, sleep
import random
from threading import Thread

"""
    Settings
"""

display = True
model_name = "Q22"
# Q14 well working


batch_size = 100  # train multiple times on one dataset

branch = [5, 5, 5]

gameover_penalty = -1000
survive_bonus = 0.1

behavior_e = 0.02

motion = False

life_duration = 0
score_duration = 0
life_list = []
score_list = []

"""
    Loading model
"""
epoch = 1
new_model = False
with open('modeldata.txt', 'r') as f:
    lines = [x.replace('\n', '') for x in f.readlines()]
    for line in lines:
        if len(line.split('\t')) != 3: continue
        modelname, current_epoch, r_history = line.split('\t')
        if modelname == model_name:
            epoch = int(current_epoch)
            Q = keras.models.load_model("./model/%s.h5" % model_name)
            score_list = [int(r_history.split(',')[-1])]
            break
        line = f.readline()
    else:
        new_model = True

if new_model:
    Q = build_model()
    Q.compile(loss="MSE", optimizer=keras.optimizers.Adam(learning_rate=0.0005))
    Q.save("./model/%s.h5" % model_name)
    with open('modeldata.txt', 'a') as f:
        f.write("%s\t%d\t%s\n" % (model_name, epoch, '0'))

#Q = keras.models.load_model("./backup/%s_%d.h5" % ("Q22", 27))
#epoch = 27 * 10


def save_model():
    global Q, epoch
    with open('modeldata.txt', 'r') as f:
        lines = [x.replace('\n', '').split('\t') for x in f.readlines()]
    with open('modeldata.txt', 'w') as f:
        for line in lines:
            if line[0] == model_name:
                score_history = list(map(int, line[2].split(',')))[:epoch - 1]
                while len(score_history) < epoch - 1:
                    score_history += [0]
                score_history += [max(score_list[-1], score_duration) if len(score_list) > 0 else 0]
                # rollback
                if False and epoch > 50 and 0.7 * max(score_history[:-20]) > max(score_list[-20:]):
                    max_ind = floor(int(np.argmax(np.asarray(score_history))) / 10)
                    Q = keras.models.load_model("./backup/%s_%d.h5" % (model_name, max_ind))
                    epoch = max_ind * 10
                    score_history = score_history[:epoch]
                    f.write("%s\t%d\t%s\n" % (model_name, epoch, ','.join(map(str, score_history))))
                    continue

                f.write("%s\t%d\t%s\n" % (model_name, epoch, ','.join(map(str, score_history))))
            else:
                f.write("%s\t%s\t%s\n" % (line[0], line[1], line[2]))
    Q.save("./model/%s.h5" % model_name)

    if epoch % 10 == 0:
        Q.save("./backup/%s_%d.h5" % (model_name, int(epoch / 10)))


"""
    =========
"""


def tick(target: PyTetris.Window):
    while target.tick():
        sleep(0.01)


window = None
agent = None
if display:
    window = PyTetris.Window()
    window.set_ghost(0)
    window.set_gravity(0)
    if motion:
        agent = PyTetris.Player(window)
        agent.set_speed(50)
        window.set_ghost(1)

S = State(10, 20)

X, Y = [], []

qvalue = Display()


def show_move(s_from, dest):
    window.set_state(s_from)
    agent.set_dest(dest)
    while agent.has_dest():
        agent.tick()
        sleep(0.01)
        if not window.tick(): break


T = Thread(target=show_move, args=(S, (0, 18, 0)))

while True:
    for batch in range(batch_size):
        t0 = time()
        s0 = S.copy()
        transition_sample = policy_greedy_e_sample(S, Q, branch[0])
        nodes = [[[x[0], x[1], x[2] if (x[1][1] >= 0 or x[1] == (-1, -1, -1)) else gameover_penalty,
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
        leaf_transitions = []
        for leaf in nodes[-1]:
            leaf_transitions = leaf[0].transitions()
            # leaf_transitions = random.sample(leaf_transitions, min(len(leaf_transitions), 10))
            assign.append(len(leaf_transitions))
            predict_input += [state_to_layer(leaf[0], x[0]) for x in leaf_transitions]

        predict_output = Q.predict(np.asarray(predict_input)).flatten()

        assign_ind = 0
        for i, leaf in enumerate(nodes[-1]):
            Q_values = predict_output[assign_ind: assign_ind + assign[i]]
            assign_ind += assign[i]
            Q_average = max(Q_values) * 0.95 + ((sum(Q_values) - max(Q_values)) / (len(Q_values) - 1)) * 0.05
            nodes[-2][leaf[1]][3].append(leaf[2] + gamma * Q_average)

        final_Q = []

        for i in reversed(range(len(nodes) - 1)):
            for node in nodes[i]:
                Q_average = max(node[3]) * 0.95 + ((sum(node[3]) - max(node[3])) / (len(node[3]) - 1)) * 0.05
                prop = node[2] + gamma * Q_average
                if i > 0:
                    nodes[i - 1][node[1]][3].append(prop)
                else:
                    final_Q.append(prop)

        for i, node in enumerate(nodes[0]):
            X.append(state_to_layer(S, node[0]))
            Y.append(final_Q[i])

        israndom = random.random() < behavior_e  # e-greedy
        a = int(np.argmax(np.asarray(final_Q))) if not israndom else random.randint(0, len(nodes[0]) - 1)

        S = nodes[0][a][0].copy()
        r = nodes[0][a][2]

        qvalue.print(
            ("[Epoch %d]\t" % epoch) + "#" * int(batch * 30 / batch_size) + "=" * (30 - int(batch * 30 / batch_size))
            + "\tTime/Step:\t%.3f" % (time() - t0)
            + "\tQ:\t%.3f" % max(final_Q)
            + "\tscore:\t%.2f" % (score_duration)
            + "\tlife:\t%d" % (life_duration)
            + "\ta:\t%d" % (a)
        )

        if r == gameover_penalty:
            S = PyTetris.State(10, 20)
            life_list.append(life_duration)
            score_list.append(score_duration)
            life_duration = 0
            score_duration = 0
        else:
            life_duration += 1
            score_duration += r

        if display:
            if motion:
                if T.is_alive():
                    T.join()
                T = Thread(target=show_move, args=(s0, nodes[0][a][1]))
                T.start()
            else:
                window.set_state(S)
            if not window.tick(): break

    qvalue.clear()

    loss = Q.train_on_batch(np.asarray(X), np.asarray(Y))
    print("[Epoch %d]" % epoch, end='\t')
    print("loss:\t%.4f" % loss, end='\t')
    if len(life_list) > 5:
        life_list = life_list[-5:]
        score_list = score_list[-5:]
    print("life:\t%.4f" % (max(life_list[-1], life_duration) if len(life_list) > 0 else 0), end='\t')
    print("score:\t%.4f" % (max(score_list[-1], score_duration) if len(score_list) > 0 else 0))
    save_model()
    X = X[int(len(X) * 0.3):]
    Y = Y[int(len(Y) * 0.3):]
    epoch += 1
