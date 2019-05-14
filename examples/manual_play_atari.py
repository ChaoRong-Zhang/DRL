import gym
from gym.utils.play import play


if __name__ == '__main__':
    env_name = 'SpaceInvaders-v0'
    env = gym.make(env_name)
    play(env, zoom=4)
