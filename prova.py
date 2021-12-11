from sklego.meta.grouped_predictor import GroupedPredictor
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklego.datasets import load_chicken

df = load_chicken(as_frame=True)

print(df.head())

# Create a binary target
X = df.drop(columns='weight')
y = np.where(df.weight>df.weight.mean(),1,0)

mod = GroupedPredictor(LogisticRegression(), groups=["diet"])

mod.fit(X,y)

mod.predict(X)
print(mod.predict_proba(X))

mod_2 = GroupedPredictor(LinearRegression(), groups=["diet"])


mod_2.fit(X, df['weight'])
print(mod_2.predict_proba(X))

# print(mod.__predict_single_group(X, "diet"))
# print(mod.predict_proba_single_group(1, X.drop(columns='diet')))
# print(mod.predict_proba_single_group(2, X.drop(columns='diet')))
# print(mod.predict_proba_groups(3, X.drop(columns='diet')))