import numpy as np

class MyLinearRegression():
    """
    Linear Regression Class
    """
    def __init__(self,
                 l1_ratio: float = 0,
                 l2_ratio: float = 0,
                 learning_rate: float = .1,
                 threshold: float = .001,
                 early_stopping_rounds: int = 100,
                 verbose: int = 0,
                 max_iter = 100000) -> None:
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.max_iter = max_iter
        self.coef_ = None
        self.best_iteration = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.insert(X, 0, 1, axis=1)
        m, n = X.shape
        self.coef_ = np.zeros(n)
        min_loss = float('inf')
        no_changes_counter = 0
        
        for i in range(self.max_iter):
            predictions = X.dot(self.coef_)
            errors = predictions - y

            gradient = X.T.dot(errors) / m
            # Additional L1 or L2 Regularization
            if self.l1_ratio > 0:
                gradient[1:] += self.l1_ratio * np.sign(self.coef_[1:])
            if self.l2_ratio > 0:
                gradient[1:] += 2 * self.l2_ratio * self.coef_[1:]

            self.coef_ = self.coef_ - self.learning_rate * gradient

            # Loss with L1 and L2 Addition
            loss = np.sum(errors ** 2) / (2 * m)
            if self.l1_ratio > 0:
                loss += self.l1_ratio * np.sum(np.abs(self.coef_[1:]))
            if self.l2_ratio > 0:
                loss += self.l2_ratio* np.sum(self.coef_[1:] ** 2)
            
            if self.verbose == 0:
                pass
            elif not i % self.verbose:
                print(f'Iteration: {i}, MSE: {loss}')

            if min_loss - loss <= self.threshold:
                no_changes_counter += 1
                if no_changes_counter == self.early_stopping_rounds:
                    self.best_iteration = i
                    return
            else:
                no_changes_counter = 0
            
            min_loss = min(min_loss, loss)
            
        self.best_iteration = self.max_iter

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.coef_)