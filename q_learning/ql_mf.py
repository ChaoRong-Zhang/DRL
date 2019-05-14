import numpy as np
import os
import logging

import gym

class q_learning_model_free:
    def __init__(self, env_name, num_s):
        """
        A class implement simple model-free Q learning method
        Args:
        """
        self.env = gym.make(env_name)
        self.q_table = {}
        
        self.num_s = num_s
        self.run_iter = 10000
        self.max_steps = 200
        self.discount_factor = 1
        # self.learning_rate = .001
        self.learning_rate_max = 1.
        self.learning_rate_min = 0.4
        self.epsilon = 0.02

        log_path = os.path.abspath(__file__).split('/')[-1].split('.')[0]
        logging.basicConfig(level=logging.DEBUG,
                            handlers=[logging.FileHandler("{0}.log".format(log_path)),
                                      logging.StreamHandler()])


    def obs_to_state(self, obs):
        interval = (self.env.observation_space.high - self.env.observation_space.low) / self.num_s
        state = (obs - self.env.observation_space.low) / interval
        return list(map(int, state))

    def inference(self):
        """
        Inference from the initial state to the end state based on the Q table
        """
        # self.env.seed(0)
        # np.random.seed(0)
        cur_obs = self.env.reset()
        total_reward = 0
        for cur_step in range(self.max_steps):
            cur_s = self.obs_to_state(cur_obs)
            cur_q_vals = []
            for a in range(self.env.action_space.n):
                if tuple([tuple(cur_s), a]) not in self.q_table:
                    self.q_table[tuple([tuple(cur_s), a])] = 0
                cur_q_vals.append(self.q_table[tuple([tuple(cur_s), a])])
            cur_a = np.argmax(cur_q_vals)
            nxt_obs, cur_reward, done, _ = self.env.step(cur_a)
            total_reward += cur_reward
            nxt_s = self.obs_to_state(nxt_obs)
            nxt_q_vals = []
            for a in range(self.env.action_space.n):
                if tuple([tuple(nxt_s), a]) not in self.q_table:
                    self.q_table[tuple([tuple(nxt_s), a])] = 0
                nxt_q_vals.append(self.q_table[tuple([tuple(nxt_s), a])])
            cur_k = tuple([tuple(cur_s), cur_a])
            if cur_k in self.q_table:
                self.q_table[cur_k] = (1 - self.learning_rate) * self.q_table[cur_k] + self.learning_rate * (cur_reward + self.discount_factor * np.max(nxt_q_vals))
            else:
                self.q_table[cur_k] = self.learning_rate * (cur_reward + self.discount_factor * np.max(nxt_q_vals))

            self.env.render()
            if done:
                break
            cur_s[:] = nxt_s
            cur_obs = np.copy(nxt_obs)

        logging.info(f'Total {cur_step} steps, {total_reward} rewards.')
        self.env.close()

    def choose_action(self, state):
        rand_val = np.random.uniform(0, 1)
        if rand_val < self.epsilon:
            action = np.random.choice(self.env.action_space.n)
        else:
            logits = []
            for a in range(self.env.action_space.n):
                if tuple([tuple(state), a]) not in self.q_table:
                    self.q_table[tuple([tuple(state), a])] = 0
                logits.append(self.q_table[tuple([tuple(state), a])])
            logits_exp = np.exp(logits)
            probs = logits_exp / np.sum(logits_exp)
            action = np.random.choice(self.env.action_space.n, p=probs)
        return action

    def run(self, mode='q_learning'):
        for i in range(self.run_iter):
            # self.env.seed(0)
            # np.random.seed(0)
            if i % 1000 == 0:
                logging.info(f'Processing in iteration {i}')
            cur_obs = self.env.reset()
            total_reward = 0
            self.learning_rate = max(self.learning_rate_min, self.learning_rate_max * (0.85 ** (i//1000)))
            for cur_step in range(self.max_steps):
                cur_s = self.obs_to_state(cur_obs)
                cur_a = self.choose_action(cur_s)
                nxt_obs, cur_reward, done, _ = self.env.step(cur_a)
                total_reward += cur_reward
                nxt_s = self.obs_to_state(nxt_obs)
                nxt_q_vals = []
                for a in range(self.env.action_space.n):
                    if tuple([tuple(nxt_s), a]) not in self.q_table:
                        self.q_table[tuple([tuple(nxt_s), a])] = 0
                    nxt_q_vals.append(self.q_table[tuple([tuple(nxt_s), a])])
                cur_k = tuple([tuple(cur_s), cur_a])
                if cur_k in self.q_table:
                    self.q_table[cur_k] = (1 - self.learning_rate) * self.q_table[cur_k] + self.learning_rate * (cur_reward + self.discount_factor * np.max(nxt_q_vals))
                else:
                    self.q_table[cur_k] = self.learning_rate * (cur_reward + self.discount_factor * np.max(nxt_q_vals))

                if done:
                    print(total_reward)
                    break
                cur_s[:] = nxt_s
                cur_obs = np.copy(nxt_obs)

        self.inference()
