"""
==============================================================
DOM Classification
==============================================================

"""

# %%
# Generate data
# -------------
#
# In order to learn good latent representations from a small dataset, we
# artificially generate more labeled data by perturbing the training data with
# linear shifts of 1 pixel in each direction.
import joblib as joblib
import pandas
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn2pmml import SelectorProxy
from sklearn2pmml.decoration import ContinuousDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn_pandas import DataFrameMapper

# path = '/tmp/dataset-15921387134184871264.csv'
# samples = pd.read_csv(path, sep=' ', header=None)
# Y = samples.iloc[0:, 0].values
# X = samples.iloc[0:, 1:].values

df = pandas.read_csv("Iris.csv")

X = df[df.columns.difference(["Species"])]
y = df["Species"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

column_preprocessor = DataFrameMapper([
    (["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], [ContinuousDomain(), StandardScaler()])
])
table_preprocessor = Pipeline([
    ("pca", PCA(n_components=3)),
    ("selector", SelectorProxy(SelectKBest(k=2)))
])
dom_raw_classifier = tree.DecisionTreeClassifier(min_samples_leaf=5)
pipeline = PMMLPipeline([
    ("columns", column_preprocessor),
    ("table", table_preprocessor),
    ("classifier", dom_raw_classifier)
])
dom_raw_classifier.fit(X_train, Y_train)

export_path = '/tmp/dom_classification.pipeline.pkl.z'
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
