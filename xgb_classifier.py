from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

def xgb_model(X_train,y_train):
    classifier=XGBClassifier(learning_rate=0.01)
    xg_parameters={
    'n_estimators':[300,400,500,700],
    'colsample_bytree':[0.8,1]
    }
    grid_search_xg=GridSearchCV(estimator=classifier,
                                param_grid=xg_parameters,
                                n_jobs=-1,
                                scoring='accuracy')
    grid_search_xg.fit(X_train,y_train)

    model=grid_search_xg.best_estimator_
    # calibrated_XG=CalibratedClassifierCV(estimator=model,method='sigmoid')
    # calibrated_XG.fit(X_train,y_train)

    return model

def make_prediction(model,X_test):
    y_pred=model.predict_proba(X_test)
    return y_pred
