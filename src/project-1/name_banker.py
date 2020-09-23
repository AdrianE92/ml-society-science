import numpy as np

class RandomBanker:
    
    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    def fit(self, X, y):
        """using navie bayes
        """
        self.data = [X, y]
        


    # set the interest rate
    def set_interest_rate(self, rate):
        self.rate = rate
        return

    def get_proba(self, action):
        self.action = action
        return action

    # Predict the probability of failure for a specific person with data x
    def predict_proba(self, x):
        return 0

    # THe expected utility of granting the loan or not. Here there are two actions:
    # action = 0 do not grant the loan
    # action = 1 grant the loan
    #
    # Make sure that you extract the length_of_loan from the
    # 2nd attribute of x. Then the return if the loan is paid off to you is amount_of_loan*(1 + rate)^length_of_loan
    # The return if the loan is not paid off is -amount_of_loan.
    def expected_utility(self, x, action):
        """x[1] is length of loan and x[4] is amount of loan
        """
        if get_proba(self, action) == 0:
            return 0 
        if get_proba(self, action) == 1:
            return x[4] * (1 + self.rate)**x[1]

    # Return the best action. This is normally the one that maximises expected utility.
    # However, you are allowed to deviate from this if you can justify the reason.
    def get_best_action(self, x):
        return np.random.choice(2,1)[0]
