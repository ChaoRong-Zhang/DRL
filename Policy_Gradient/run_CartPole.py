import gym
from REINFORCE import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
)

for i_episode in range(3000):
    print("episode:", i_episode)

    observation = env.reset()
    steps=0
    while steps<10000:
        steps+=1
        if RENDER==True:
            env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        RL.store_transition(observation, action, reward)

        if done:
            # ep_rs_sum = sum(RL.ep_rs)

            # if 'running_reward' not in globals():
            #     running_reward = ep_rs_sum
            # else:
            #     running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            # if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            # print("episode:", i_episode, "  reward:", int(running_reward))
            RL.learn()
            if i_episode % 10==0:
                RENDER = True
            else:
                RENDER = False


            break

        observation = observation_
env.close()
