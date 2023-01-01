"""
==============================================================
DOM Classification
==============================================================


"""

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

# %%
# Generate data
# -------------
#
# In order to learn good latent representations from a small dataset, we
# artificially generate more labeled data by perturbing the training data with
# linear shifts of 1 pixel in each direction.
import joblib as joblib
import pandas as pd
from sklearn import linear_model
from sklearn.base import clone
from sklearn.model_selection import train_test_split

path = '/tmp/dataset-15921387134184871264.csv'
samples = pd.read_csv(path, sep=' ', header=None)
Y = samples.iloc[0:, 0].values
X = samples.iloc[0:, 1:].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

logistic = linear_model.LogisticRegression(solver="newton-cg", tol=1)
logistic.C = 6000
# Training the Logistic regression classifier directly on the pixel
dom_raw_classifier = clone(logistic)
dom_raw_classifier.C = 100.0
dom_raw_classifier.fit(X_train, Y_train)

export_path = '/tmp/dom_logistic_classification.pipeline.pkl.z'
# Storing the fitted PMMLPipeline object in pickle data format:
joblib.dump(dom_raw_classifier, export_path, compress=9)

# %%
# Evaluation
# ----------

from sklearn import metrics

classifier = joblib.load(export_path)
# %%
Y_pred = classifier.predict(X_test)
print(
    "Logistic regression using raw features:\n%s\n"
    % (metrics.classification_report(Y_test, Y_pred))
)

# %%
# path = '/tmp/1.csv'
# samples = pd.read_csv(path, sep=' ', header=None)
# YY_test = samples.iloc[0:, 0].values
# XX_test = samples.iloc[0:, 1:].values
# YY_pred = dom_raw_classifier.predict(XX_test)
# print(
#     "Logistic regression to predict given file using raw features:\n%s\n"
#     % (metrics.classification_report(YY_test, YY_pred))
# )
