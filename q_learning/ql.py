import numpy as np


class q_learning:
    def __init__(self, num_s, a_spc, r_spc):
        self.num_s = num_s
        if len(a_spc)!=num_s:
            raise ValueError('Shape mismatch between state and action space')
        self.a_spac = a_spc
        if num_s!=len(r_spc):
            raise ValueError('Shape mismatch between state and reward space')
        for i in range(len(a_spc)):
            if action[i] != len(r_spc[i]):
                raise ValueError(f'Shape mismatch between action {i} and reward space')
