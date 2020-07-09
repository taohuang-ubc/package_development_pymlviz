"""
Created on March 3, 2020

@author: Anas Muhammad

Implementation of model_comparison_table in the
pymlviz package.
"""
import pandas as pd
from sklearn.base import is_classifier, is_regressor


def model_comparison_table(X_train, y_train, X_test, y_test, **kwargs):
    """
    Takes in scikit learn ML models
    of the same family (regression
    or classification) and the train
    test data then outputs a table
    comparing the scores for
    different models.

    Parameters:
    ------------
    X_train : pd.DataFrame/np.ndarray
        Training dataset without labels.

    y_train : np.ndarray
        Training labels.

    X_test : pd.DataFrame/np.ndarray
        Test dataset without labels.

    y_test : np.ndarray
        Test labels.

    **kwargs :
        Models assigned with meaningful
        variable names.

    Returns:
    ------------
    pd.DataFrame
        Dataframe object consisting of
        models and comparison metrics.

    Example:
    ------------
    >>> from sklearn.datasets import make_classification
    >>> from pymlviz.model_comparison_table  import model_comparison_table
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.svm import SVC
    >>> from sklearn.model_selection import train_test_split
    >>> import pandas as pd
    >>> syn_data_cf = make_classification(n_samples=1000, n_classes=4,
    >>>                                 n_informative=12)
    >>> tts_cf = train_test_split(pd.DataFrame(syn_data_cf[0]),
    >>>                         syn_data_cf[1],
    >>>                         test_size=0.33, random_state=42)
    >>> X_train, X_test, y_train, y_test = tts_cf
    >>> lr_cf = LogisticRegression().fit(X_train, y_train)
    >>> svm_cf = SVC().fit(X_train, y_train)
    >>> model_comparison_table(X_train, y_train, X_test, y_test,
    >>>     lr_model=lr_cf, svm_model=svm_cf)
    >>> print(model_comparison_table(X_train, y_train,
    >>>     X_test, y_test,
    >>>     lr_model=lr_cf, svm_model=svm_cf))
    """
    try:
        # check if all regression or all classification
        regression_check = True
        classification_check = True
        # classification check
        for model_type in kwargs.values():
            regression_check &= is_regressor(model_type)
            classification_check &= is_classifier(model_type)

        assert (classification_check | regression_check), \
            "Please enter all regression or classification models"

        # create dataframe skeleton for model
        df_results = pd.DataFrame({"model_name": [],
                                   "train_score": [],
                                   "test_score": []})

        # loop through models specified by user
        for model in kwargs:
            # compute values for results table
            train_score = kwargs[model].score(X_train, y_train)
            test_score = kwargs[model].score(X_test, y_test)
            model_name = model

            # create temporary results table
            df_res = pd.DataFrame({"model_name": [model_name],
                                   "train_score": [train_score],
                                   "test_score": [test_score]})

            # update results table
            df_results = df_results.append(df_res, ignore_index=True)

        # return dataframe
        return df_results

    except AssertionError as Error:
        return Error
