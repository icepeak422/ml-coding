import numpy as np

class GaussianNaiveBayes:
    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model.
        
        Parameters:
        - X: ndarray of shape (N, D) — feature matrix
        - y: ndarray of shape (N,) — class labels
        """
        self.classes = np.unique(y)
        self.class_stats = {}

        for cls in self.classes:
            X_c = X[y == cls]
            self.class_stats[cls] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0) + 1e-9,  # avoid division by zero
                'prior': X_c.shape[0] / X.shape[0]
            }

    def _gaussian_log_likelihood(self, X, mean, var):
        """
        Compute log-likelihood of data X under a Gaussian distribution.
        """
        return -0.5 * np.sum(np.log(2 * np.pi * var) + ((X - mean) ** 2) / var, axis=1)

    def predict(self, X):
        """
        Predict the class labels for input X.
        """
        log_probs = []

        for cls in self.classes:
            stats = self.class_stats[cls]
            log_likelihood = self._gaussian_log_likelihood(X, stats['mean'], stats['var'])
            log_prior = np.log(stats['prior'])
            log_probs.append(log_likelihood + log_prior)

        return self.classes[np.argmax(np.stack(log_probs, axis=1), axis=1)]