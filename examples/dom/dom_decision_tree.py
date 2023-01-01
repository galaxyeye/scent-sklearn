"""
==============================================================
HTML DOM Classification
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
import pandas
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

path = '/tmp/dataset-15921387134184871264.csv'

df = pandas.read_csv(path, sep=' ')

X = df[df.columns.difference(["T"])]
y = df["T"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

dom_raw_classifier = tree.DecisionTreeClassifier(min_samples_leaf=5)
pipeline = PMMLPipeline([
    ("classifier", dom_raw_classifier)
])
pipeline.fit(X_train, Y_train)
# pipeline.verify(X_train.sample(n=100))

export_path = '/tmp/dom_decision_tree.pkl.z'
# Storing the fitted PMMLPipeline object in pickle data format:
joblib.dump(pipeline, export_path, compress=9)

sklearn2pmml(pipeline, "/tmp/dom_decision_tree.pmml", with_repr=True)

# %%
# Evaluation
# ----------

classifier = joblib.load(export_path)
# %%
Y_pred = classifier.predict(X_test)
print(
    "Decision tree using raw features:\n%s\n"
    % (metrics.classification_report(Y_test, Y_pred))
)
