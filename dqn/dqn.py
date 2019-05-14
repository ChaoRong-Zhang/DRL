import numpy as np
import gym


class DQN:
    def __init__(self, env_name):
        self.env = gym.make(env_name)

    def run(self):
        play(self.env, zoom=4)




test_class = DQN('SpaceInvaders-v0')
test_class.run()
