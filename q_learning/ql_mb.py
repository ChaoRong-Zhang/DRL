import numpy as np
import os
import logging


class q_learning_model_based:
    def __init__(self, num_s, a_spc, r_spc, nxt_s_spc, s_start, s_end):
        """
        A class implement simple model-based Q learning method
        Args:
            num_s: number of states
            a_spc: list of number of actions for each state.
            r_spc: 2 dimentional list of rewards for each state index and action index.
            nxt_s_spc: 3 dimentional list recording the probability to each state based
                on current state index and action index.
            s_state: index of state to s_start
            s_end: index of state to stop
        Returns:
            The ID of the checkpoint from which the actor was resumed, or None
            if the actor should restart from the beginning.
        """

        self.num_s = num_s
        if len(a_spc)!=num_s:
            raise ValueError('Shape mismatch between state and action space')
        self.a_spc = a_spc
        if num_s!=len(r_spc):
            raise ValueError('Shape mismatch between state and reward space')
        for s in range(num_s):
            if a_spc[s] != len(r_spc[s]):
                raise ValueError(f'Shape mismatch between action {s} and reward space')
        self.r_spc = r_spc
        for s in range(num_s):
            for a in range(a_spc[s]):
                if num_s != len(nxt_s_spc[s][a]):
                    raise ValueError(f'Shape mismatch for probability to next state for state {s}, action {a}')
        self.nxt_s_spc = nxt_s_spc
        self.s_start = s_start
        self.s_end = s_end
        self.q_table = {}

        # properties
        self.run_iter = 1000
        self.test_iter = 10
        self.max_steps = 10000
        self.discount_factor = 0.9
        self.learning_rate = 0.1
        self.epsilon = 0.1

        log_path = os.path.abspath(__file__).split('/')[-1].split('.')[0]
        logging.basicConfig(level=logging.DEBUG,
                            handlers=[logging.FileHandler("{0}.log".format(log_path)),
                                      logging.StreamHandler()])

    def get_nxt_state(self, ind_s, ind_a):
        """
        Get the next state index based on current state index and action index
        Args:
            ind_s: current state index
            ind_a: current action index
        Returns:
            next state index
        """
        prob_nxt_s = self.nxt_s_spc[ind_s][ind_a]
        rand_val = np.random.uniform(0, 1)
        acc_val = 0
        for i in range(self.num_s):
            acc_val += prob_nxt_s[i]
            if rand_val < acc_val:
                return i

    def inference(self):
        """
        Inference from the initial state to the end state based on the Q table
        """
        for i in range(self.test_iter):
            cur_s = self.s_start
            cur_step = 0
            total_reward = 0
            while cur_s != self.s_end and cur_step < self.max_steps:
                cur_q_vals = []
                for a in range(self.a_spc[cur_s]):
                    if tuple([cur_s, a]) not in self.q_table:
                        self.q_table[tuple([cur_s, a])] = 0
                    cur_q_vals.append(self.q_table[tuple([cur_s, a])])
                action = np.argmax(cur_q_vals)
                nxt_s = self.get_nxt_state(cur_s, action)
                cur_reward = self.r_spc[cur_s][action]
                logging.debug(f'Step {cur_step}: from state {cur_s} to state {nxt_s}, with reward {cur_reward}')
                total_reward += cur_reward
                cur_s = nxt_s
                cur_step += 1
            if cur_step < self.max_steps:
                logging.info(f'Inference Trial {i}: Success! Total {cur_step} steps, {total_reward} rewards.')
            else:
                logging.info(f'Inference Trial {i}: Failure! Final state is in state {cur_s}')

    def run(self):
        """
        Run multiple trials from initial state to the end state. Learn the best strategy and update on Q table
        """
        for i in range(self.run_iter):
            if i % 100 == 0:
                logging.info(f'Processing in iteration {i}')
            cur_s = self.s_start
            cur_step = 0
            while cur_s != self.s_end and cur_step < self.max_steps:
                rand_val = np.random.uniform(0, 1)
                if rand_val < self.epsilon:
                    action = np.random.choice(self.a_spc[cur_s])
                else:
                    logits = []
                    for a in range(self.a_spc[cur_s]):
                        if tuple([cur_s, a]) not in self.q_table:
                            self.q_table[tuple([cur_s, a])] = 0
                        logits.append(self.q_table[tuple([cur_s, a])])
                    logits_exp = np.exp(logits)
                    probs = logits_exp / np.sum(logits_exp)
                    action = np.random.choice(self.a_spc[cur_s], p=probs)
                nxt_s = self.get_nxt_state(cur_s, action)
                nxt_q_vals = []
                for a in range(self.a_spc[nxt_s]):
                    if tuple([nxt_s, a]) not in self.q_table:
                        self.q_table[tuple([nxt_s, a])] = 0
                    nxt_q_vals.append(self.q_table[tuple([nxt_s, a])])
                cur_k = tuple([cur_s, action])
                if cur_k in self.q_table:
                    self.q_table[cur_k] = (1 - self.learning_rate) * self.q_table[cur_k] + self.learning_rate * (self.r_spc[cur_s][action] + self.discount_factor * np.max(nxt_q_vals))
                else:
                    self.q_table[cur_k] = self.learning_rate * (self.r_spc[cur_s][action] + self.discount_factor * np.max(nxt_q_vals))

                cur_s = nxt_s
                cur_step += 1
        self.inference()
