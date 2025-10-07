# Iris + RandomForest – clean, reproducible, and informative

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

RSEED = 77
np.random.seed(RSEED)

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target                    # use the provided integer labels (0,1,2)
class_names = iris.target_names    # ["setosa","versicolor","virginica"]

# Quick peek (optional)
print(X.head().to_string(index=False))

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RSEED
)
print(f"\nTrain size: {len(X_train)}   Test size: {len(X_test)}")

# Model
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=RSEED,
    n_jobs=-1
)

# Fit & predict
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.3f}")


# Confusion matrix with readable labels
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print("\nConfusion matrix:")
print(cm_df)

