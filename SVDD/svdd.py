import numpy as np
from sklearn.base import BaseEstimator
from cvxopt import matrix, solvers


class SVDD(BaseEstimator):
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.dot(X, X.T)

        # Define the quadratic programming problem
        P = matrix(K)
        q = matrix(-np.ones((n_samples, 1)))
        G = matrix(np.vstack((np.eye(n_samples), -np.eye(n_samples))))
        h = matrix(
            np.vstack((self.C * np.ones((n_samples, 1)), np.zeros((n_samples, 1))))
        )
        A = matrix(np.ones((1, n_samples)))
        b = matrix(1.0)

        # Solve the quadratic programming problem
        solvers.options["show_progress"] = False
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution["x"])

        # Compute the center of the sphere
        self.center_ = np.dot(alphas.T, X).reshape(-1)

        # Compute the radius of the sphere
        distances = np.linalg.norm(X - self.center_, axis=1)
        self.radius_ = np.max(distances)

        return self

    def predict(self, X):
        n_samples, n_features = X.shape

        # Compute the distance from the center of the sphere
        distances = np.linalg.norm(X - self.center_, axis=1)

        # Predict the labels of X
        predictions = np.ones(n_samples, dtype=int)
        predictions[distances > self.radius_] = -1

        return predictions


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate some data
    np.random.seed(0)
    X = np.random.randn(100, 2)
    X[80:] += 4

    # Fit the SVDD model
    model = SVDD(C=0.1)
    model.fit(X[:80])

    # Predict the labels of X
    y_pred = model.predict(X)

    # Plot the results
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.scatter(model.center_[0], model.center_[1], c="red", marker="x")
    circle = plt.Circle(model.center_, model.radius_, color="red", fill=False)
    plt.gca().add_artist(circle)
    plt.xlim(-4, 8)
    plt.ylim(-4, 8)
    plt.gca().set_aspect("equal")
    plt.show()
