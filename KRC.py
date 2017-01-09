import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.kernel_ridge import KernelRidge

class KernelRidgeClassifier(KernelRidge):
    """Kernel Ridge Classification model"""
    def predict(self, X):
        return np.sign(super(KernelRidgeClassifier, self).predict(X))

    def score(self, X, y):
        return metrics.accuracy_score(y, self.predict(X))
