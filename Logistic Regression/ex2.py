

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

RSEED = 77
np.random.seed(RSEED)

# -------------------------
# Load data
# -------------------------
CSV_PATH = "wine_quality.csv"
  
df = pd.read_csv(CSV_PATH)

print(df.head().to_string(index=False))
#%%
# -------------------------
# Data Preprocessing
# -------------------------


features = df.drop(columns=['Unnamed: 0','quality'],axis =1).copy() 
cat_cols = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
for col in cat_cols:
    features[col] = pd.factorize(features[col])[0]

class_target = df['quality']

class_target_conv,class_names = pd.factorize(class_target)

#%%
# -------------------------
# # Train/test split 
# -------------------------


x_train, x_test, y_train, y_test = train_test_split(
    features, class_target, test_size=0.25, random_state=RSEED
)


# -------------------------
#  Model Building and Prediction
# -------------------------
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=RSEED,
    n_jobs=-1
)

# Fit & predict
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.3f}")


# -------------------------
#  Evaluation, and Analysis
# -------------------------


# Confusion matrix with readable labels
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print("\nConfusion matrix:")
print(cm_df)

# Cross-validated accuracy (optional but useful)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RSEED)
cv_scores = cross_val_score(clf, features, class_target_conv, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"\n5-fold CV accuracy: mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}")

# Feature importance (sorted)
fi = pd.Series(clf.feature_importances_, index=features.columns).sort_values(ascending=False)
print("\nFeature importances:")
print(fi)











# cm = confusion_matrix(y_test, y_pred_lr)
# cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
# print("\nConfusion matrix:")

# print(cm_df)
