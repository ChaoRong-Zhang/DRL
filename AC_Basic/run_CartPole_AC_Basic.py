import gym
import tensorflow as tf
from AC_Basic import Actor, Critic
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 0  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

sess= tf.Session()

actor = Actor(
    sess=sess,
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate = 0.005,
)
critic = Critic(
    sess=sess,
    n_features = env.observation_space.shape[0],
    learning_rate = 0.01,
)
sess.run(tf.global_variables_initializer())

for i_episode in range(3000):
    print("episode:", i_episode)

    observation = env.reset()
    steps=0
    while steps<10000:
        steps+=1
        if RENDER==True:
            env.render()

        action = actor.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        if done:
            reward=-20
        td_error = critic.learn(observation, reward, observation_)

        actor.store_transition(observation, action, float(td_error))

        if done or steps>10000:

            ep_rs_sum = sum(actor.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "step:", steps, "  reward:", float(running_reward))

            actor.learn()
            # if i_episode % 10==0:
            #     RENDER = True
            # else:
            #     RENDER = False


            break

        observation = observation_
env.close()
