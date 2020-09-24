import pandas

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
df = pandas.read_csv('./data/credit/german.data', sep=' ',
                     names=features+[target])
df['repaid'] = df['repaid'].map(mapping)

#df = pandas.read_csv('D_valid.csv', sep=' ',
#                    names=features+[target])
#df = pa

import matplotlib.pyplot as plt
numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)
encoded_features = list(filter(lambda x: x != target, X.columns))

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
import name_banker  #this is a random banker
import random_banker
decision_maker = name_banker.NameBanker()
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
    decision_maker.set_interest_rate(interest_rate)
    decision_maker.fit(X_train, y_train)
    Ui, Ri = test_decision_maker(X_test, y_test, interest_rate, decision_maker)
    utility += Ui
    investment_return += Ri

print("Average utility:", utility / n_tests)
print("Average return on investment:", investment_return / n_tests)


