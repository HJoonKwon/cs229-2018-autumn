import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    x_train, t_train = util.load_dataset(train_path,
                                         label_col='t',
                                         add_intercept=True)
    x_test, t_test = util.load_dataset(test_path,
                                       label_col='t',
                                       add_intercept=True)
    model_t = LogisticRegression()
    model_t.fit(x_train, t_train)
    preds = model_t.predict(x_test)
    preds = preds >= 0.5
    acc = np.sum(preds == t_test) / t_test.shape[0]
    print(f"Accuracy on test set for 2.(c) is {acc}")
    util.plot(x_test, t_test, model_t.theta, save_path='output/p02c.png')

    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    x_train, y_train = util.load_dataset(train_path,
                                         label_col='y',
                                         add_intercept=True)
    x_test, y_test = util.load_dataset(test_path,
                                       label_col='y',
                                       add_intercept=True)
    model_y = LogisticRegression()
    model_y.fit(x_train, y_train)
    preds = model_y.predict(x_test)
    preds = preds >= 0.5
    acc = np.sum(preds == y_test) / y_test.shape[0]
    print(f"Accuracy on test set for 2.(d) is {acc}")
    util.plot(x_test, y_test, model_y.theta, save_path='output/p02d.png')

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e

    x_val, y_val = util.load_dataset(valid_path,
                                     label_col='y',
                                     add_intercept=True)
    preds = model_y.predict(x_val)
    alpha = np.sum(preds) / preds.shape[0]

    correction = 1 + np.log(2 / alpha - 1) / model_y.theta[0]
    util.plot(x_test, t_test, model_y.theta, 'output/p02e.png', correction)

    preds = model_y.predict(x_test) / alpha
    preds = preds >= 0.5
    acc = np.sum(preds == t_test) / t_test.shape[0]
    print(f"Accuracy on test set for 2.(e) is {acc}")

    # *** END CODER HERE
