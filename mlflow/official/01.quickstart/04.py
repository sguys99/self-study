# 특정 training run을 로드하여 추론하기

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

model = mlflow.sklearn.load_model("runs:/6af2a8f18ced4970809460fa8b9d705b/model") # id 뒤에 model을 입력해야 가능..
predictions = model.predict(X_test)
print(predictions)