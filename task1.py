"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
# You can use this space to define any helper functions that you need

def vectorized_kl_divergence(p_values, q_values):
    epsilon = 1e-6  # Small constant to prevent division by zero and logarithmic issues
    
    q_values = np.maximum(q_values, epsilon)  # Ensure q_values are greater than or equal to epsilon
    
    term1 = (1 - p_values) * np.log((1 - p_values + epsilon) / (1 - q_values + epsilon))
    term2 = p_values * np.log((p_values + epsilon)/ q_values)
    
    kl_values = np.where(p_values == 0.0, term1, np.where(p_values == 1.0, term2, term1 + term2))
    
    return kl_values

def get_q_kl(values, bounds):
    q = np.zeros(len(values))

    a = values  # lower bound for q
    b = np.ones(len(values))  # upper bound for q

    epsilon = 1e-4
    while np.any(b - a > epsilon):
        c = (a + b) / 2
        kl_c = vectorized_kl_divergence(values, c)

        mask = kl_c <= bounds
        a = np.where(mask, c, a)
        b = np.where(mask, b, c)

    q = (a + b) / 2
    return q

def get_beta(alpha, beta):
    samples = np.random.beta(alpha + 1, beta + 1)
    return samples

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.ucb = np.ones(num_arms)
        self.counts = np.ones(num_arms)
        self.values = np.zeros(num_arms) #contains the average of the reward for each arm
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if(len(np.where(self.ucb == np.max(self.ucb))) > 1):
            return np.random.choice(np.where(self.ucb == np.max(self.ucb)))
        else:
            return np.argmax(self.ucb)
        # raise NotImplementedError
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        total_counts = np.sum(self.counts)
        exploration_term = np.sqrt(2 * np.log(total_counts) / self.counts)
        self.ucb = self.values + exploration_term   
        # raise NotImplementedError
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.ucb_kl = np.zeros(num_arms)
        self.counts = np.ones(num_arms)
        self.values = np.zeros(num_arms) #contains the average of the reward for each arm
        self.bounds = np.zeros(num_arms) #contains the rhs of the inequality, defined for each arm
        self.c = 0 #will be used in finding q
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if(len(np.where(self.ucb_kl == np.max(self.ucb_kl))) > 1):
            return np.random.choice(np.where(self.ucb_kl == np.max(self.ucb_kl)))
        else:
            return np.argmax(self.ucb_kl)        
        # raise NotImplementedError
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        total_counts = np.sum(self.counts)
        self.bounds = (np.log(total_counts) + self.c*np.log(np.log(total_counts)))/self.counts
        self.ucb_kl = get_q_kl(self.values ,self.bounds)
        # raise NotImplementedError
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.success = np.ones(num_arms) 
        self.failures = np.ones(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        samples = get_beta(self.success, self.failures)
        if(len(np.where(samples == np.max(samples))) > 1):
            return np.random.choice(np.where(samples == np.max(samples)))
        else:
            return np.argmax(samples) 
        # raise NotImplementedError
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if(reward == 1):
            self.success[arm_index] +=1
        else:
            self.failures[arm_index] +=1
        # raise NotImplementedError
        # END EDITING HERE
