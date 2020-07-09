from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import numpy as np


def plot_confusion_m(model, X_test, y_test, labels=None, title=None):
    """
    Takes in a trained model with X and y
    values to produce a confusion matrix
    visual. If predicted_y array is passed in,
    other evaluation scoring metrics such as
    Recall, and precision will also be produced.

    Parameters:
    ------------
    model : model instance
        A trained classifier

    X_test : pd.DataFrame/np.ndarray
        Test dataset without labels.

    y_test : np.ndarray
        Test labels.

    labels : list, default=None
        The labels of the confusion matrix

    title : String, default=None
        Title of the confusion matrix

    Returns:
    ------------
    display : matplotlib visual

    Example:
    ------------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>> from pymlviz.plot_confusion_m import plot_confusion_m
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
    >>>               columns=iris['feature_names'] + ['target'])
    >>> X = iris_df.drop(columns=['target'])
    >>> y = iris_df[['target']]
    >>> X_train, X_valid, y_train, y_valid = train_test_split(X,
    >>> y.to_numpy().ravel(), test_size=0.2, random_state=123)
    >>> svm = SVC()
    >>> svm.fit(X_train, y_train)
    >>> plot_confusion_m(svm, X_valid, y_valid)
    """

    if not isinstance(X_test, pd.DataFrame) and not \
            isinstance(X_test, np.ndarray):
        raise Exception("Sorry, X_test should be a "
                        "pd.DataFrame or np.ndarray.")

    if not isinstance(y_test, np.ndarray):
        raise Exception("Sorry, y_test should be a np.ndarray.")

    if (isinstance(X_test, pd.DataFrame) and
        not np.issubdtype(X_test.to_numpy().dtype, np.number)) or \
            (isinstance(X_test, np.ndarray) and
             not np.issubdtype(X_test.dtype, np.number)):
        raise Exception("Sorry, all elements "
                        "in X_test should be numeric.")

    if not np.issubdtype(y_test.dtype, np.number):
        raise Exception("Sorry, all elements "
                        "in y_valid should be numeric.")

    if y_test.shape[0] != X_test.shape[0]:
        raise Exception("Sorry, X_test and y_test "
                        "should have the same number of rows.")

    confusion_matrix = plot_confusion_matrix(model, X_test, y_test,
                                             display_labels=labels,
                                             values_format='d')
    if title is None:
        confusion_matrix.ax_.set_title('Confusion Matrix')
    else:
        confusion_matrix.ax_.set_title(title)

    return confusion_matrix.figure_
