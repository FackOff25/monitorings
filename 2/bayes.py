import numpy as np

class NaiveBayesG:
    def fit(self, X, y):
        self.classes, classesVolumes = np.unique(y, return_counts=True)
        self.priories = classesVolumes / len(y)

        self.X_mus = np.array([np.mean(X[y == c], axis=0) for c in self.classes])
        self.X_sigmas = np.array([np.std(X[y == c], axis=0) for c in self.classes])

    def gaussBayesFormula(self, x, mu, sigma):
        return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2 / (2 * sigma ** 2)))

    def predict(self, X):
        possibilities = np.array([self.gaussBayesFormula(x, self.X_mus, self.X_sigmas) for x in X])
        posteriors = self.priories * np.prod(possibilities, axis=2)
        return np.array([self.classes[pred] for pred in np.argmax(posteriors, axis=1)])