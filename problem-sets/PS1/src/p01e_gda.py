import numpy as np

import util
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    preds = preds >= 0.5
    acc = np.sum(preds == y_test) / y_test.shape[0]
    print(f"Accuracy on testset is {acc}")
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        y = y.reshape((-1, 1))

        self.theta = np.zeros((n + 1, 1))

        phi = 1 / m * np.sum(y == 1)
        mu0 = np.sum((y == 0) * x, axis=0) / np.sum(y == 0)
        mu1 = np.sum((y == 1) * x, axis=0) / np.sum(y == 1)

        cov = 1 / m * (x - (y == 0) * mu0 -
                       (y == 1) * mu1).T @ (x - (y == 0) * mu0 -
                                            (y == 1) * mu1)

        mu0 = mu0.reshape(-1, 1)
        mu1 = mu1.reshape(-1, 1)
        theta = np.linalg.inv(cov) @ (mu1 - mu0)

        theta0 = 0.5 * (mu0 +
                        mu1).T @ np.linalg.inv(cov) @ (mu0 - mu1) - np.log(
                            (1 - phi) / phi)
        theta0 = np.squeeze(theta0)

        self.theta[0] = theta0
        self.theta[1:] = theta

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        z = x @ self.theta
        htheta = 1 / (1 + np.exp(-z))
        return htheta.squeeze()
        # *** END CODE HERE
