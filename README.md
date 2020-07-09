## pymlviz 

![](https://github.com/UBC-MDS/pymlviz/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/pymlviz/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/pymlviz) ![Release](https://github.com/UBC-MDS/pymlviz/workflows/Release/badge.svg) [![Documentation Status](https://readthedocs.org/projects/pymlviz-final/badge/?version=latest)](https://pymlviz-final.readthedocs.io/en/latest/?badge=latest)

Visualization package for ML models. 

> This package contains four functions to allow users to conveniently plot various visualizations as well as compare performance of different classifier models. The purpose of this package is to reduce time spent on developing visualizations and comparing models, to speed up the model creation process for data scientists. The four functions will perform the following tasks: 
> 1.  Compare the performance of various models 
> 2.  Plot the confusion matrix based on the input data
> 3.  Plot train/validation errors vs. parameter values
> 4.  Plot the ROC curve and calculate the AUC 

|Contributors|GitHub Handle|
|------------|-------------|
|Anas Muhammad| [anasm-17](https://github.com/anasm-17)|
|Tao Huang|[taohuang-ubc](https://github.com/taohuang-ubc)|
|Fanli Zhou|[flizhou](https://github.com/flizhou)|
|Mike Chen|[miketianchen](https://github.com/miketianchen)|

### Installation:

```
pip install -i https://test.pypi.org/simple/ pymlviz
```

### Features
| Function Name | Input | Output | Description |
|-------------|-----|------|-----------|
|model_comparison_table| List of model, X_train, y_train, X_test, y_test, scoring option | Dataframe of model score| Takes in a list of models and the train test data then outputs a table comparing the scores for different models.|
|plot_confusion_matrix | Model, X_train, y_train, X_test, y_test, predicted_y  | Confusion Matrix Plot, Dataframe of various scores (Recall, F1 and etc)| Takes in a trained model with X and y values to produce a confusion matrix visual. If predicted_y array is passed in, other evaluation scoring metrics such as Recall, and precision will also be produced.|
|plot_train_valid_error| model_name, X_train, y_train, X_valid, y_valid, param_name, param_vec |Train/validation errors vs. parameter values plot| Takes in a model name, train/validation data sets, a parameter name and a vector of parameter values and then plots train/validation errors vs. parameter values.|
|plot_roc|model, X_valid, y_valid|ROC plot| Takes in a fitted model, the validation set(X_valid) and the validation set labels(y_valid) and plots the ROC curve. The ROC curve also produces AUC score.|

### Alignment with Python Ecosystems

For some of our functions, there are not existing packages that implement the exact same functionality in Python. Most of these functions helps to show insights about machine learning models conveniently.

The comparisons between new functions and existing functions are:

| New functions | Existing Packages/Functions |
|-------------|-----|
|plot_confusion_matrix| [Sklearn's classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) | 


### Dependencies

- python 3.7.3 with packages:
  - pandas == 1.0.1
  - numpy == 1.18.1
  - matplotlib == 3.1.3
  - python-semantic-release == 4.10.0
  - scikit-learn == 0.22.2
  - altair == 3.2.0

### Usage

#### model_comparison_table()

```
from sklearn.datasets import make_classification
from pymlviz.model_comparison_table  import model_comparison_table
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
syn_data_cf = make_classification(n_samples=1000, n_classes=4,
                                n_informative=12)
tts_cf = train_test_split(pd.DataFrame(syn_data_cf[0]),
                        syn_data_cf[1],
                        test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = tts_cf
lr_cf = LogisticRegression().fit(X_train, y_train)
svm_cf = SVC().fit(X_train, y_train)
model_comparison_table(X_train, y_train, X_test, y_test,
    lr_model=lr_cf, svm_model=svm_cf)
print(model_comparison_table(X_train, y_train,
    X_test, y_test,
    lr_model=lr_cf, svm_model=svm_cf))
    
>>>     model_name  train_score  test_score
      0   lr_model     0.689552    0.709091
      1  svm_model     0.907463    0.836364
```

#### plot_roc()

```
from pymlviz.plot_roc import plot_roc
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from pymlviz.plot_roc import plot_roc
from sklearn.datasets import load_breast_cancer

# generate data from breast cancer dataset
breast_cancer = load_breast_cancer(return_X_y=True)
X, y = breast_cancer
X_train_breast = X[:400]
y_train_breast = y[:400]
X_valid_breast = X[400:569]
y_valid_breast = y[400:569]

# fit model
svc_proba = SVC(probability=True)
svc_proba.fit(X_train_breast, y_train_breast)

# plot roc curve
plot_roc(svc_proba, X_valid_breast, y_valid_breast)

>>> <plot_output>
```

#### plot_train_valid_error()

```
from pymlviz.plot_train_valid_error import plot_train_valid_error
import pandas as pd
import numpy as np
import altair as alt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# use the iris data for unittest
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])

X = iris_df.drop(columns=['target'])
y = iris_df[['target']]

X_train, X_valid, y_train, y_valid = train_test_split(X,
                                                      y.to_numpy().ravel(),
                                                      test_size=0.2,
                                                      random_state=123)
plot_train_valid_error('knn',
                       X_train, y_train,
                       X_valid, y_valid,
                       'n_neighbors', range(1, 50))
>>> <plot_output>
```

### plot_confusion_m()

```
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from pymlviz.plot_confusion_m import plot_confusion_m
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])

X = iris_df.drop(columns=['target'])
y = iris_df[['target']]
X_train, X_valid, y_train, y_valid = train_test_split(X,
          y.to_numpy().ravel(), test_size=0.2, random_state=123)

svm = SVC()
svm.fit(X_train, y_train)
plot_confusion_m(svm, X_valid, y_valid)

>>> <plot_output>
```

### Documentation
The official documentation is hosted on Read the Docs: <https://pymlviz-final.readthedocs.io/en/latest/>

### Credits
This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage). 
