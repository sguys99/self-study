import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

#mlflow.atutolog를 사용하면 자동으로 파라미터, 메트릭, artifacts를 로깅한다.
# https://mlflow.org/docs/latest/tracking.html#automatic-logging

db = load_diabetes()

mlflow.autolog()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

predictions = rf.predict(X_test)

# 실행하면 mlruls 디렉토리 안에 실험 결과가 저장됨
# 그리고 mlflow ui를 실행하면 UI에서 실험결과를 확인 가능

# autolog를 지원하지 않는 라이브러리를 사용한다면  다음을 사용하면 된다.
# parameters : 상수, mlflow.log_param, mlflow.logparams
# Metrics: 동작중에 업데이트 되는 값들(ex : accuracy), mlflow.log_metric
# Artifacts: 동작중에 생성되는 파일들(ex: model weights), mlflow.log_artifacts, mlflow.log_image, mlflow.log_text

