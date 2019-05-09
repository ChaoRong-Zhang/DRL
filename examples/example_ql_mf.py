import sys
sys.path.append('./../')
from q_learning.ql_mf import q_learning_model_free


test_model = q_learning_model_free(env_name='MountainCar-v0', num_s=20)
test_model.run(mode='q_learning')
# test_model.run(mode='sarsa')
