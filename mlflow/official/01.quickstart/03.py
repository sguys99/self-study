# mlflow에 모델 저장하기
# mlflow의 Model 디렉토리에는 다음이 저장된
# MLModel 파일 : yaml 포멧이며 모델의 flavor, dependency, signature, 중요한 메타정보 등이 저장됨
# 모델 런타임 환경을 재 생성하는데 필요한 것들 : conda.yaml
# 선택적으로 입력 예제....

import mlflow
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# autologging을 사용하는 경우, mlflow는 모델이나 코드가 생성하는 모들 것을 기록한다.
#또는 수동으로 mlflow.{library_module_name}.log_model을 호출하여 수동으로 로깅할 수도 있다.
# 그리고 run ID를 콘솔에 직접 출력할 수도 있다.

with mlflow.start_run() as run:
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)
    
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)
    
    predictions = rf.predict(X_test)
    print(predictions)
    
    signature = infer_signature(X_test, predictions)
    mlflow.sklearn.log_model(rf, "model", signature=signature)
    
    print("Run ID: {}".format(run.info.run_id))