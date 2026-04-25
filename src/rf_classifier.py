from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

def rf_model(X_train,y_train,league):
    """Here is the creation of the RF Classifier Model. This function returns the model"""
    if league.lower()=='laliga':
        rf_classifier=RandomForestClassifier(criterion='entropy',class_weight='balanced')
    else:
        rf_classifier=RandomForestClassifier(criterion='entropy')
    tscv=TimeSeriesSplit(n_splits=5)
    parameters = {
    'n_estimators': [100, 150,200],
    'max_depth': [5, 8, 12],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 5, 10],
    'max_features':['sqrt',0.8,None]
    }
    grid_search=GridSearchCV(estimator=rf_classifier,
                            param_grid=parameters,
                            n_jobs=-1,
                            cv=tscv,
                            scoring='accuracy')

    grid_search.fit(X_train,y_train)

    model=grid_search.best_estimator_
    return model

def make_prediction(model,X_test):
    y_pred=model.predict_proba(X_test)
    return y_pred
