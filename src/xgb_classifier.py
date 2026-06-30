from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path 
import sys
project_root = str(Path(__file__).resolve().parent.parent)
print(f"Project Root: {project_root}")
if project_root not in sys.path:
    sys.path.append(project_root)
from mlflow_script import get_safe_cv_splits

def xgb_model(X_train,y_train,league):

    if league.lower()=='bundesliga':
        classifier=XGBClassifier(
            random_state=42,
            n_estimators=120,
            subsample=0.8,
            min_child_weight=3,
            max_depth=3,
            learning_rate=0.01,
            colsample_bytree=0.8
        )
    elif league.lower()=='laliga':
        classifier=XGBClassifier(
            random_state=42,
            learning_rate=0.01,
            n_estimators=100,
            subsample=0.8,
            min_child_weight=3,
            max_depth=3,
            colsample_bytree=0.7
        )
    elif league.lower()=='premier league':
        classifier=XGBClassifier(
            random_state=42,
            learning_rate=0.01,
            max_depth=3,
            min_child_weight=3,
            n_estimators=70,
            subsample=0.8
        )
    elif league.lower()=='greek super league':
        classifier=XGBClassifier(
            random_state=42,
            n_estimators=110,
            max_depth=3,
            min_child_weight=2,
            subsample=0.8,
            learning_rate=0.01,
            colsample_bytree=0.8
        )
    else:
        raise ValueError(f"League: {league} is not supported")
    
    classifier.fit(X_train,y_train)
    cv=get_safe_cv_splits(X_train,y_train,5)
    calibrated_XG=CalibratedClassifierCV(estimator=classifier,cv=cv,method='sigmoid')
    calibrated_XG.fit(X_train,y_train)

    return calibrated_XG

def make_prediction(model,X_test):
    y_pred=model.predict_proba(X_test)
    return y_pred
