import numpy as np


class DL_Model:
    def __init__(self, _model):
        self.queue = []
        self.index = []
        self.result = []
        self.model = _model
        self.ret_index = 0
        self.ret_at = 0

    def put(self, _x):
        self.queue += _x
        self.index.append(len(_x))

    def evaluate(self):
        self.result = self.model.predict(np.asarray(self.queue))

    def get(self):
        ret = self.result[self.ret_at:self.ret_at + self.index[self.ret_index]]
        self.ret_at += self.index[self.ret_index]
        self.ret_index += 1
        return ret

    def clear(self):
        self.queue = []
        self.index = []
        self.result = []
        self.ret_index = 0
        self.ret_at = 0


def state_to_layer(s0, s1):
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
