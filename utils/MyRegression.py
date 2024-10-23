import numpy as np
import matplotlib.pyplot as plt

class MyRegression():
    """
    Regression Class for Linear and Logistic Regression with L1/L2 Regularization

    This class supports both linear and logistic regression, with options for L1 (Lasso) or L2 (Ridge) regularization.

    Parameters
    ----------
    type : str, default='linear'
        Specifies the type of regression to perform. Options are:
        - 'linear' for Linear Regression
        - 'logistic' for Logistic Regression

    l1_ratio : float, default=0
        Regularization strength for Lasso (L1) regression. A value of 0 disables L1 regularization.
        Cannot be used simultaneously with l2_ratio > 0.

    l2_ratio : float, default=0
        Regularization strength for Ridge (L2) regression. A value of 0 disables L2 regularization.
        Cannot be used simultaneously with l1_ratio > 0.

    learning_rate : float, default=0.1
        The step size for gradient descent optimization.

    threshold : float, default=0.001
        The tolerance level for determining whether the new loss is considered better than the previous one.

    early_stopping_rounds : int, default=100
        The number of iterations to continue training without improvement before stopping early.

    verbose : int, default=0
        Controls the verbosity level. A value of 0 means no information will be printed during training.

    max_iter : int, default=100000
        The maximum number of iterations allowed for the training process.

    """
    def __init__(self, type: str = 'linear', # 'linear' or 'logistic'
                 l1_ratio: float = 0,
                 l2_ratio: float = 0,
                 learning_rate: float = .1,
                 threshold: float = .001,
                 early_stopping_rounds: int = 100,
                 verbose: int = 0,
                 max_iter = 100000) -> None:
        if type not in ['linear', 'logistic']:
            raise ValueError(f"Invalid regression_type '{type}'. Expected 'linear' or 'logistic'.")
        self.type = type
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        if (l1_ratio > 0) and (l2_ratio > 0):
            raise ValueError(f"Couldn't use L1 and L2 simultaneuosly.")
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.max_iter = max_iter
        self.coef_ = None
        self.best_iteration_ = None
        self.loss_values_ = np.array([])

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Private Function to compute Sigmoid in .fit() method 
        """
        return 1 / (1 + np.exp(-z))
    
    def _loss_computation(self, predictions: np.ndarray, y: np.ndarray, m: int) -> float:
        """
        Private Function to compute loss in .fit() method 
        """  
        if self.type == 'linear':
            errors = predictions - y
            loss = np.sum(errors ** 2) / (2 * m)
        elif self.type == 'logistic':
            loss = -np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions)) / m

        if self.l1_ratio > 0:
            loss += self.l1_ratio * np.sum(np.abs(self.coef_[1:]))
        if self.l2_ratio > 0:
            loss += (self.l2_ratio / 2) * np.sum(self.coef_[1:] ** 2)

        return loss

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Main .fit() method
        """
        X = np.insert(X, 0, 1, axis=1)
        m, n = X.shape
        self.coef_ = np.zeros(n)
        min_loss = float('inf')
        no_changes_counter = 0
        
        for i in range(self.max_iter):
            predictions = X.dot(self.coef_)
            if self.type == 'logistic':
                predictions = self._sigmoid(predictions)
            errors = predictions - y

            # Counting Gradient
            gradient = X.T.dot(errors) / m
            # Additional L1 or L2 Regularization
            if self.l1_ratio > 0:
                gradient[1:] += self.l1_ratio * np.sign(self.coef_[1:])
            if self.l2_ratio > 0:
                gradient[1:] += self.l2_ratio * self.coef_[1:]

            self.coef_ = self.coef_ - self.learning_rate * gradient

            # Loss Computation
            loss = self._loss_computation(predictions, y, m)
            self.loss_values_ = np.append(self.loss_values_, loss)
            
            if self.verbose == 0:
                pass
            elif not i % self.verbose:
                print(f'Iteration: {i}, MSE: {loss}')

            if min_loss - loss <= self.threshold:
                no_changes_counter += 1
                if no_changes_counter == self.early_stopping_rounds:
                    self.best_iteration_ = i
                    return
            else:
                no_changes_counter = 0
            
            min_loss = min(min_loss, loss)
            
        self.best_iteration_ = self.max_iter

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Main .predict() Method
        """
        X = np.insert(X, 0, 1, axis=1)  # Add intercept term
        predictions = X.dot(self.coef_)
        if self.type == 'logistic':
            predictions = self._sigmoid(predictions)
        return predictions
