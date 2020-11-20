import numpy as np
import pandas
np.random.seed(42)

def default_reward_function(action, outcome):
    return -0.1 * (action!= 0) + outcome

def test_policy(generator, policy, reward_function, T):
    print("Testing for ", T, "steps")
    policy.set_reward(reward_function)
    u = 0
    for t in range(T):
        x = generator.generate_features()
        a = policy.recommend(x)
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        u += r
        policy.observe(x, a, y)
        #print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
    return u
"""
features = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values
"""
features = pandas.read_csv('../../data/medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('../../data/medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('../../data/medical/historical_Y.dat', header=None, sep=" ").values

observations = features[:, :128]
labels = features[:,128] + features[:,129]*2

import data_generation
#import random_recommender
#policy_factory = random_recommender.RandomRecommender
#import reference_recommender
#policy_factory = reference_recommender.HistoricalRecommender

### Importing a LogisticRegression-recommender ###
import lr_recommender
policy_factory = lr_recommender.LogisticRegressionRecommender

def print_test():
    """ Testing with the same number of treatments """
    print("---- Testing with only two treatments ----")
    print("Setting up simulator")
    #generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat")
    print("Setting up policy")
    
    policy = policy_factory(2, 2)
    policy.set_reward(default_reward_function)

    ## Fit the policy on historical data first
    print("Fitting historical data to the policy")
    policy.fit_treatment_outcome(features, actions, outcome)
    lr_utility = policy.estimate_utility(features, None, None, policy) / features.shape[0]
    hist_utility = policy.estimate_utility(features, actions, outcome) / features.shape[0]
    print(lr_utility)
    print(hist_utility)
    """
    ## Run an online test with a small number of actions
    print("Running an online test")
    n_tests = 1000
    result = test_policy(generator, policy, default_reward_function, n_tests)
    print("Total reward:", float(result))
    print("Final analysis of results")
    policy.final_analysis()
    """
print_test()

# ## First test with the same number of treatments
# print("--- Testing with an additional experimental treatment and 126 gene silencing treatments ---")
# print("Setting up simulator")
# generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat")
# print("Setting up policy")
# policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
# ## Fit the policy on historical data first
# print("Fitting historical data to the policy")
# policy.fit_treatment_outcome(features, actions, outcome)
# ## Run an online test with a small number of actions
# print("Running an online test")
# n_tests = 1000
# result = test_policy(generator, policy, default_reward_function, n_tests)
# print("Total reward:", result)
# print("Final analysis of results")
# policy.final_analysis()
