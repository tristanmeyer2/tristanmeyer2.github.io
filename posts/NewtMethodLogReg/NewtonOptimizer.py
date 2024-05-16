import torch

class LinearModel:

    def __init__(self):
        self.w = None 
        self.w_prev = None

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        if self.w_prev is None:
            self.w_prev = torch.zeros(X.size()[1])

        return torch.mv(X,self.w) 

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        scores = self.score(X)
        return torch.where(scores > 0, 1.0, 0)
    
class LogisticRegression(LinearModel):
    def loss(self, X, y):
        """
        Compute the empirical risk of the model using the score function. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        """
        def sig(s):
            calc = 1 / (1 + torch.exp(-s)) 
            calc[calc == 1] = 0.9999999
            calc[calc == 0] = 0.0000001
            return calc

        def empirical_risk(s, y, X):
            return (-(y @ sig(s).log().T + (1 - y)@(1-sig(s)).log().T))/X.shape[0]
        
        y_hat = self.score(X)

        return empirical_risk(y_hat, y, X)
    
    def grad(self, X, y):
        """
        Computing the gradient of the empirical risk
        """
        def sig(s):
            return 1 / (1 + torch.exp(-s))
        s = self.score(X) 
        return (((sig(s))-y)@X)/X.shape[0]
    
    def hessian(self, X):
        """
        Computing the Hessian Matrix, the matrix of second derivatives of L, using feature matrix X. 
        """
        def sig(s):
            return 1 / (1 + torch.exp(-s))
        s = self.score(X)
        diag = sig(s)*(1-sig(s))
        D = torch.zeros(diag.shape[0], diag.shape[0])
        D[range(len(D)), range(len(D))] = diag
        return X.T@D@X

    
class NewtonOptimizer():
    def __init__(self, model):
        self.model = model 

    def step(self, X, y, alpha):
        """
        Compute one step of the logistic update using the feature matrix X, target vector y,
        and an alpha value. 
        """
        grad = self.model.grad(X,y)
        hessian = self.model.hessian(X)
        self.model.w = self.model.w - alpha*(hessian.inverse()@grad)



        
