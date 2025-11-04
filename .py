from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
          callbacks=[EarlyStopping(rounds=10)])
print("✅ Works fine now!")
