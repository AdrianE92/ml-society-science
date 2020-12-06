import numpy as np
import pandas

import thompson_bandit
import data_generation

def reward_function(action, outcome):
    return -0.1 * action + outcome

def test_policy_additional(generator, policy,  T, reward_function=reward_function):
    print("Additional treatments testing for ", T, "steps")
    policy.set_reward(reward_function)
    u = 0
    not_placebo = 0 #counting the people not given placebo
    number_of_treatments = 129
    actioncount = np.zeros(number_of_treatments)
    for t in range(T):
        x = generator.generate_features()
        a = policy.recommend(x)
        actioncount[a] += 1
        if (a > 0): 
            not_placebo += 1
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        u += r
        policy.observe(x, a, y)
        #print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
        
    return [u/T, not_placebo, actioncount]

if __name__ == "__main__":
    prior_a = [2,2] + [1]*127
    prior_b = [5,2] + [1]*127
    policy = thompson_bandit.ThompsonBandit(129, 2, prior_a, prior_b)
    generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat")
    result1 = test_policy_additional(generator, policy, 1000)
   
    #print("10 most used treatments:")
    b = np.argsort(-result1[2])
    print("Treatment ", b, "used in patients ", result[2][b], " number of times. ")
    #print("Treatment:  %5d %5d %5d %5d %5d %5d %5d %5d %5d %5d" % (b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9]))
    #r = result1[2][b]
    #print("Times used: %5d %5d %5d %5d %5d %5d %5d %5d %5d %5d" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]))


    T = 500000
    #policy = thompson_bandit.ThompsonBandit(129, 2, prior_a, prior_b)
    result2 = test_policy_additional(generator, policy, T)
    #print("10 most used treatments:")
    b = np.argsort(-result2[2])
    print("Treatment ", b, "used in patients ", result[2][b], " number of times. ")
    #print("Treatment:  %6d %6d %5d %5d %5d %5d %5d %5d %5d %5d" % (b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9]))
    #r = result1[2][b]
    #print("Times used: %6d %6d %5d %5d %5d %5d %5d %5d %5d %5d" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]))


    print("\nThompson  (20 000): utility = %f" %result1[0])
    print("Thompson (200 000): utility = %f" %result2[0])




"""
features = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
labels = features[:,128] + features[:,129]*2

import data_generation
import random_recommender
policy_factory = random_recommender.RandomRecommender
import reference_recommender
#policy_factory = reference_recommender.HistoricalRecommender

## First test with the same number of treatments
print("---- Testing with only two treatments ----")

print("Setting up simulator")
generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat")
print("Setting up policy")
policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
## Fit the policy on historical data first
print("Fitting historical data to the policy")
policy.fit_treatment_outcome(features, actions, outcome)
## Run an online test with a small number of actions
print("Running an online test")
n_tests = 1000
result = test_policy(generator, policy, default_reward_function, n_tests)
print("Total reward:", result)
print("Final analysis of results")
policy.final_analysis()

## First test with the same number of treatments
print("--- Testing with an additional experimental treatment and 126 gene silencing treatments ---")
print("Setting up simulator")
generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat")
print("Setting up policy")
policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
## Fit the policy on historical data first
print("Fitting historical data to the policy")
policy.fit_treatment_outcome(features, actions, outcome)
## Run an online test with a small number of actions
print("Running an online test")
n_tests = 1000
result = test_policy(generator, policy, default_reward_function, n_tests)
print("Total reward:", result)
print("Final analysis of results")
policy.final_analysis()

"""



