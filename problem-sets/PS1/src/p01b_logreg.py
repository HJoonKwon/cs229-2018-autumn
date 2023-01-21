import numpy as np

from src import util
from src.linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    preds = preds >= 0.5
    acc = np.sum(preds == y_test) / y_test.shape[0]
    print(f'Accuracy on testset is {acc}')

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape

        y = y.reshape((-1, 1))

        if not self.theta:
            self.theta = np.zeros((n, 1))

        for i in range(self.max_iter):
            theta_old = np.array(self.theta, copy=True)
            z = x @ self.theta
            htheta = 1 / (1 + np.exp(-z))

            H = 1/m * (x * htheta * (1 - htheta)).T @ x
            Hinv = np.linalg.inv(H)

            dtheta = 1/m * x.T @ (htheta - y)

            self.theta -= self.step_size * Hinv @ dtheta
            loss = -1/m * (y.T @ np.log(htheta) + (1-y).T @ np.log(1 - htheta))
            if self.verbose and i % 10 == 0:
                print(f'Loss @ iter{i} ={loss}')

            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break

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
        htheta = np.squeeze(htheta)
        return htheta

        # *** END CODE HERE ***
