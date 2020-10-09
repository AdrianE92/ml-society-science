import pandas
import numpy as np
#Helper function to map the values of repaid
def mapping(x):
    if x == 2:
        x = 0
    else:
        x = 1
    return x

## Set up for dataset
features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'

df = pandas.read_csv('../../data/credit/german.data', sep=' ',
                     names=features+[target])
df['repaid'] = df['repaid'].map(mapping)

#df = pandas.read_csv('../../data/credit/german.data', sep=' ', names=features+[target])
#df = pandas.read_csv('../../data/credit/D_valid.csv', sep=' ', names=features+[target])


import matplotlib.pyplot as plt
numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
quantitative_features_2 = []
X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)
for i in X.columns:
    if i not in numerical_features and i != 'repaid':
        quantitative_features_2.append(i)

encoded_features = list(filter(lambda x: x != target, X.columns))

def qua_noise(X):
    for i in quantitative_features_2:
        #print(X[i].size)
        w = np.random.choice([0, 1], size=(len(X), len(quantitative_features_2)), p=[0.7, 0.3])
        noise = np.random.choice(np.unique(X[i]), size=w.sum())
        
        X[i] = (X[i] + w) % 2
        
    return X


#Create noise using differential privacy through laplace
#We implement a coin-toss to randomize what data becomes noisy.
def laplace_func(X):
    X_noise = X.copy()
    epsilon = 0.01
    n = len(X)

    if np.random.random() > 0.5:
        for i in numerical_features:
            k = len(i)
            M = (X[i].max()-X[i].min())
            l = (M*epsilon)/k
            w = np.random.laplace(scale=l, size=n)    
            X_noise[i] += w
    return X_noise

## Test function
def test_decision_maker(X_test, y_test, interest_rate, decision_maker):
    n_test_examples = len(X_test)
    utility = 0

    ## Example test function - this is only an unbiased test if the data has not been seen in training
    total_amount = 0
    total_utility = 0
    decision_maker.set_interest_rate(interest_rate)
    for t in range(n_test_examples):
        action = decision_maker.get_best_action(X_test.iloc[t])
        good_loan = y_test.iloc[t] # assume the labels are correct
        duration = X_test['duration'].iloc[t]
        amount = X_test['amount'].iloc[t]
        # If we don't grant the loan then nothing happens
        if (action==1):
            if (good_loan != 1):
                utility -= amount
            else:    
                utility += amount*(pow(1 + interest_rate, duration) - 1)
        total_utility += utility
        total_amount += amount
    return utility, total_utility/total_amount


## Main code


### Setup model
import anadma_banker  #this is a random banker
import random_banker
decision_maker = anadma_banker.AnadmaBanker()
#decision_maker = random_banker.RandomBanker()
#import aleksaw_banker
#decision_maker = aleksaw_banker.AlexBanker()

interest_rate = 0.05

### Do a number of preliminary tests by splitting the data in parts
from sklearn.model_selection import train_test_split
n_tests = 100
utility = 0
investment_return = 0
for iter in range(n_tests):
    X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
    X_train_noise = laplace_func(X_train)
    X_test_noise = laplace_func(X_test)
    X_train_noise = qua_noise(X_train_noise)
    X_test_noise = qua_noise(X_test_noise)
    
    decision_maker.set_interest_rate(interest_rate)
    decision_maker.fit(X_train_noise, y_train)
    Ui, Ri = test_decision_maker(X_test_noise, y_test, interest_rate, decision_maker)
    utility += Ui
    investment_return += Ri

print("Average utility:", utility / n_tests)
print("Average return on investment:", investment_return / n_tests)


