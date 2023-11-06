"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the FaultyBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon, fault): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np

# START EDITING HERE
def get_beta(alpha, beta):
    samples = np.random.beta(alpha + 1, beta + 1)
    return samples

# You can use this space to define any helper functions that you need
# END EDITING HERE

class FaultyBanditsAlgo:
    def __init__(self, num_arms, horizon, fault):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        self.fault = fault # probability that the bandit returns a faulty pull
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
            self.success[arm_index] += 1
        else:
            self.failures[arm_index] +=1
        # raise NotImplementedError
        #END EDITING HERE

