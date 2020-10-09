import pandas
import numpy as np
import matplotlib.pyplot as plt

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

numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
quantitative_features_2 = []
X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)
for i in X.columns:
    if i not in numerical_features and i != 'repaid':
        quantitative_features_2.append(i)

encoded_features = list(filter(lambda x: x != target, X.columns))

def qua_noise(X):
    w = np.random.choice([0, 1], size=(len(X), len(quantitative_features_2)), p=[0.7, 0.3])
    X[quantitative_features_2] = (X[quantitative_features_2] + w) % 2
    return X

def add_noise(X, numerical_features, categorical_features):
    #coinflip for categorical variables
    epsilon = 0.1
    k = np.shape(X)[1]

    flip_fraction = 1/ (1  + np.exp(epsilon/k))

    X_noise = X.copy()
    for t in list(X_noise.index):
         for c in X_noise.columns:
            # We can use the same random response mechanism for all binary features
            if any(c.startswith(i) for i in categorical_features):
                w = np.random.choice([0, 1], p=[1 - flip_fraction, flip_fraction])
                X_noise.loc[t,c] = (X_noise.loc[t,c] + w) % 2
            # For numerical features, it is different. The scaling factor should depend on k, \epsilon, and the sensitivity of that particular attribite. In this case, it's simply the range of the attribute.
            if any(c.startswith(i) for i in numerical_features):
                # calculate the range of the attribute and add the laplace noise to the original data
                M = np.max(X.loc[:,c]) - np.min(X.loc[:,c])
                l = M*k/(epsilon)
                w = np.random.laplace(0, l)   
                X_noise.loc[t,c] += w 
    return X_noise   
#Create noise using differential privacy through laplace
#We implement a coin-toss to randomize what data becomes noisy.
def laplace_func(X):
    X_noise = X.copy()
    epsilon = 0.1
    n = np.shape(X)[1]
    for i in numerical_features:
        if np.random.random() > 0.5:
 
            M = (X[i].max()-X[i].min())
            l = (M*epsilon)/n
            w = np.random.laplace(0, l)    
            X_noise[i] += w

    return X_noise


def add_noise(X_train, X_test):
    X_train_noise = laplace_func(X_train)
    X_test_noise = laplace_func(X_test)
    X_train_noise = qua_noise(X_train_noise)
    X_test_noise = qua_noise(X_test_noise)
    return X_train_noise, X_test_noise


## Test function ##
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

interest_rate = 0.05

### Do a number of preliminary tests by splitting the data in parts
from sklearn.model_selection import train_test_split
for i in range(5):
    n_tests = 100
    utility = 0
    investment_return = 0
    for iter in range(n_tests):
        X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
        X_train_noise, X_test_noise = add_noise(X_train, X_test)
        
        
        decision_maker.set_interest_rate(interest_rate)
        decision_maker.fit(X_train_noise, y_train)
        Ui, Ri = test_decision_maker(X_test_noise, y_test, interest_rate, decision_maker)
        utility += Ui
        investment_return += Ri

    print("Average utility:", utility / n_tests)
    print("Average return on investment:", investment_return / n_tests)


