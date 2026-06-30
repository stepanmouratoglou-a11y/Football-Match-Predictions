from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV 
from pathlib import Path 
import sys
project_root = str(Path(__file__).resolve().parent.parent)
print(f"Project Root: {project_root}")
if project_root not in sys.path:
    sys.path.append(project_root)
from mlflow_script import get_safe_cv_splits

def rf_model(X_train,y_train,league):
    """Here is the creation of the RF Classifier Model. This function returns the model"""
    if league.lower()=='laliga':
        rf_classifier=RandomForestClassifier(criterion='gini',
                                             n_estimators=45,
                                             max_depth=5,
                                             min_samples_split=8,
                                             random_state=42)
    elif league.lower()=='premier league':
        rf_classifier=RandomForestClassifier(criterion='log_loss',
                                             n_estimators=70,
                                             max_depth=5,
                                             max_features='sqrt',
                                             min_samples_leaf=10,
                                             min_samples_split=4,
                                             random_state=42)
    elif league.lower()=='bundesliga':
        rf_classifier=RandomForestClassifier(random_state=42,
                                             n_estimators=45,
                                             criterion='log_loss',
                                             max_depth=5,
                                             min_samples_split=10)
    elif league.lower()=='greek super league':
        rf_classifier=RandomForestClassifier(random_state=42,
                                             criterion='entropy',
                                             max_depth=5,
                                             min_samples_split=2,
                                             n_estimators=45)
    else:
        raise ValueError(f"League '{league}' is not supported.")    
    cv=get_safe_cv_splits(X_train,y_train,5)
    model=CalibratedClassifierCV(estimator=rf_classifier,cv=cv,method='sigmoid')
    model.fit(X_train,y_train)
    return model

def make_prediction(model,X_test):
    y_pred=model.predict_proba(X_test)
    return y_pred
