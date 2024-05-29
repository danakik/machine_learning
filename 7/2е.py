import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, C=1.0, gamma=1.0, kernel='rbf'):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for iter in range(self.n_iters):
            updates = 0
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - self.C * np.dot(x_i, y[idx]))
                    self.b -= self.lr * self.C * y[idx]
                    updates += 1
            if updates == 0:
                print(f'vse na {iter}')
                break
            print(f'Итерация {iter}: весы - {updates}')

    def decision_function(self, X):
        if self.kernel == 'linear':
            return np.dot(X, self.w) - self.b
        elif self.kernel == 'rbf':
            n_samples = X.shape[0]
            K = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    K[i, j] = np.exp(-self.gamma * np.linalg.norm(X[i] - X[j])**2)
            return np.dot(K, self.w) - self.b

    def predict(self, X):
        approx = self.decision_function(X)
        return np.sign(approx)

def plot_hyperplane(X, y, w, b, ax=None, title=None):
    if ax is None:
        ax = plt.gca()

    def decision_boundary(x, w, b, offset=0):
        return (-w[0] * x - b + offset) / w[1]

    ax.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])
    
    x1_1 = decision_boundary(x0_1, w, b)
    x1_2 = decision_boundary(x0_2, w, b)

    x1_1_m = decision_boundary(x0_1, w, b, 1)
    x1_2_m = decision_boundary(x0_2, w, b, 1)
    
    x1_1_p = decision_boundary(x0_1, w, b, -1)
    x1_2_p = decision_boundary(x0_2, w, b, -1)

    ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')
    ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k--')
    ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k--')

    if title:
        ax.set_title(title)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
data = pd.read_csv(url, header=None, names=columns)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = np.where(y == 0, -1, 1)
X = (X - X.mean(axis=0)) / X.std(axis=0)

C_values = [0.1, 0.5, 1.0]
gamma_values = [0.1, 0.5, 1.0]

svm_models = []
for C in C_values:
    for gamma in gamma_values:
        svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000, C=C, gamma=gamma, kernel='rbf')
        svm.fit(X, y)
        svm_models.append((svm, C, gamma))

plt.figure(figsize=(15, 10))
for i, (svm, C, gamma) in enumerate(svm_models):
    plt.subplot(len(C_values), len(gamma_values), i+1)
    plot_hyperplane(X, y, svm.w, svm.b, title=f'SVM Decision Boundary (C={C}, gamma={gamma})')
plt.show()
