import sys
sys.path.append('./../')
from q_learning.ql_mb import q_learning_model_based


num_s = 6
a_spc = [4, 3, 1, 2, 2, 1]
r_spc = [
    [51, -5, -20, 50],
    [-5, -20, -1],
    [-20],
    [-2, 0],
    [-1, -2],
    [0],
]
nxt_s_spc = [
    [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]],
    [[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1]],
    [[0, 0, 0, 1, 0, 0]],
    [[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]],
    [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
    [[0, 0, 0, 0, 0, 1]],
]
s_start = 0
s_end = 5

test_model = q_learning_model_based(num_s, a_spc, r_spc, nxt_s_spc, s_start, s_end)
test_model.run()
