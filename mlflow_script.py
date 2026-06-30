import sys
import importlib.abc
# from importlib.resources.abc import Traversable
# if not hasattr(importlib.abc, "Traversable"):
#     import importlib.resources.abc
#     importlib.abc.Traversable = importlib.resources.abc.Traversable

import os
import numpy as np
import pandas as pd
import mlflow.sklearn
import mlflow
from sklearn.model_selection import TimeSeriesSplit, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, log_loss

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.Preprocessing import preprocess


def get_safe_cv_splits(X, y, n_splits=5):
    """Generates TimeSeries splits but skips any folds that don't contain all 3 classes."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    valid_splits = []
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    
    for train_idx, test_idx in tscv.split(X):
        unique_classes = np.unique(y.iloc[train_idx].dropna())
        
        if len(unique_classes) == 3:
            valid_splits.append((train_idx, test_idx))
        else:
            print(f"      [!] Skipping tiny inner fold: Only contained outcomes {unique_classes}")
            
    if not valid_splits:
        split_point = int(len(X) * 0.8)
        train_idx = np.arange(split_point)
        test_idx = np.arange(split_point, len(X))
        valid_splits.append((train_idx, test_idx))
        
    return valid_splits


def load_data(league):
    leagues = ['Premier League', 'La Liga', 'Bundesliga', 'Greek Super League']
    if league not in leagues:
        raise ValueError(f"Enter a valid league. Options are: {leagues}")
        
    paths = {
       'Premier League': 'datasets/Cleaned_Data/premier_league_cleaned.csv',
       'La Liga': 'datasets/Cleaned_Data/laliga_cleaned.csv',
       'Bundesliga': 'datasets/Cleaned_Data/bundesliga_cleaned.csv',
       'Greek Super League': 'datasets/Cleaned_Data/greek_super_league_cleaned.csv'
    }
    print(f'Preparing {league} data...')
    dataset = pd.read_csv(paths[league])
    X = dataset.drop(columns=['FTR','HomeTeam','AwayTeam'])
    y = dataset['FTR'].map({'H':0,'D':1,'A':2})

    return X, y


def train_xgboost(X_train, y_train):
    print("Tuning XGBoost Classifier...")
    safe_cv = get_safe_cv_splits(X_train, y_train, n_splits=3)
    
    xgb = XGBClassifier(random_state=42, eval_metric="mlogloss",num_class=3,objective='multi:softprob',tree_method='hist')
    grid_params = {
        'n_estimators': [100,110,120],
        'max_depth': [3, 5, 6],
        'learning_rate': [0.01, 0.05],
        'min_child_weight':[1,2,3],
        'subsample':[0.8,1],
        'colsample_bytree':[0.7,0.8]
    }
    custom_scorer = make_scorer(log_loss, greater_is_better=False, response_method="predict_proba", labels=[0, 1, 2])
    grid = GridSearchCV(estimator=xgb, 
                        param_grid=grid_params, 
                        cv=safe_cv, 
                        scoring=custom_scorer, 
                        n_jobs=-1,
                        verbose=False)
    grid.fit(X_train, y_train)
    best_xgb = grid.best_estimator_
    
    classifier = CalibratedClassifierCV(estimator=best_xgb,cv=safe_cv,method="sigmoid")
    classifier.fit(X_train, y_train)
    return classifier, grid.best_params_


def train_random_forest(X_train, y_train):
    print("Tuning Random Forest Classifier...")
    safe_cv = get_safe_cv_splits(X_train, y_train, n_splits=3)
    
    rf = RandomForestClassifier(random_state=42)
    grid_params = {
        "n_estimators": [50,60,70],
        "max_depth": [5, 8, 12],
        "criterion": ['log_loss'],
        "min_samples_split": [4, 6,8, 10],
        'min_samples_leaf':[1,5,10,15,20],
        'max_features':['sqrt','log2']
    }

    custom_scorer = make_scorer(log_loss, greater_is_better=False, response_method="predict_proba", labels=[0, 1, 2])

    grid = GridSearchCV(
        estimator=rf,
        param_grid=grid_params,
        cv=safe_cv,
        scoring=custom_scorer,
        n_jobs=-1,
        verbose = False
    )
    grid.fit(X_train, y_train)

    classifier = grid.best_estimator_

    model=CalibratedClassifierCV(estimator=classifier,cv=safe_cv,method='sigmoid')
    model.fit(X_train,y_train)
    return model, grid.best_params_


def main():
    leagues = ['Premier League', 'La Liga', 'Bundesliga', 'Greek Super League']
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Football_Match_Prediction_Pipeline")

    for league in leagues:
        print(f"\nProcessing {league}")
        X, y = load_data(league)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

        model_performance = {
            "Random Forest": {"accuracies": [], "losses": [], "best_params": None},
            "XGBoost Calibrated": {"accuracies": [], "losses": [], "best_params": None}
        }
        trusted_types = [
            'sklearn.calibration._CalibratedClassifier', 
            'sklearn.calibration._SigmoidCalibration', 
            'xgboost.core.Booster', 
            'xgboost.sklearn.XGBClassifier'
        ]

        print(f"\nStarting Cross-Validation Process for {league}...")
        
        safe_outer_cv = get_safe_cv_splits(X_train, y_train, n_splits=5)
        
        for fold, (train_idx, test_idx) in enumerate(safe_outer_cv): 
            print(f"--- Processing Fold {fold + 1} / {len(safe_outer_cv)} ---")
            
            X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]
            
            rf_model, rf_params = train_random_forest(X_tr, y_tr)
            rf_preds = rf_model.predict(X_te)
            rf_probs = rf_model.predict_proba(X_te)
            model_performance["Random Forest"]["accuracies"].append(accuracy_score(y_te, rf_preds))
            model_performance["Random Forest"]["losses"].append(log_loss(y_te, rf_probs, labels=[0,1,2]))
            model_performance["Random Forest"]["best_params"] = rf_params
            
            xgb_model, xgb_params = train_xgboost(X_tr, y_tr)
            xgb_preds = xgb_model.predict(X_te)
            xgb_probs = xgb_model.predict_proba(X_te)
            model_performance["XGBoost Calibrated"]["accuracies"].append(accuracy_score(y_te, xgb_preds))
            model_performance["XGBoost Calibrated"]["losses"].append(log_loss(y_te, xgb_probs, labels=[0,1,2]))
            model_performance["XGBoost Calibrated"]["best_params"] = xgb_params
        
        with mlflow.start_run(run_name=f"TimeSeriesCV_{league.replace(' ', '_')}") as parent_run:
            mlflow.log_param("league", league)
            mlflow.log_param("outer_cv_splits", len(safe_outer_cv))
            
            best_mean_loss = float("inf")
            champion_run_id = None
            champion_model_name = None
            
            for model_name, perf in model_performance.items():
                with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                    mean_acc = np.mean(perf["accuracies"])
                    std_acc = np.std(perf["accuracies"])
                    mean_loss = np.mean(perf["losses"])
                    std_loss = np.std(perf["losses"])

                    print(f"\n{model_name} Cross-Validation Performance:")
                    print(f"  Mean Accuracy: {mean_acc:.2%} (±{std_acc:.2%})")
                    print(f"  Mean Log Loss: {mean_loss:.2f} (±{std_loss:.2f})")

                    mlflow.log_metric("mean_accuracy", mean_acc)
                    mlflow.log_metric("std_accuracy", std_acc)
                    mlflow.log_metric("mean_log_loss", mean_loss)
                    mlflow.log_metric("std_log_loss", std_loss)

                    print(f"Retraining final {model_name} on full X_train...")
                    
                    if model_name == 'Random Forest':
                        final_model, final_params = train_random_forest(X_train, y_train)
                        mlflow.log_params({f"final_rf_{k}": v for k, v in final_params.items()})
                        mlflow.sklearn.log_model(final_model, "model", skops_trusted_types=trusted_types)
                        
                    elif model_name == 'XGBoost Calibrated':
                        final_model, final_params = train_xgboost(X_train, y_train)
                        mlflow.log_params({f"final_xgb_{k}": v for k, v in final_params.items()})
                        mlflow.sklearn.log_model(final_model, "model", skops_trusted_types=trusted_types)

                    if mean_loss < best_mean_loss:
                        best_mean_loss = mean_loss
                        champion_run_id = child_run.info.run_id
                        champion_model_name = model_name

            print(f"\nThe best model for {league} is {champion_model_name}")
            registry_name = f"Football_Predictor_{league.replace(' ', '_')}"
            model_uri = f"runs:/{champion_run_id}/model"
            
            print(f"Registering best model under name '{registry_name}'...")
            model_details = mlflow.register_model(model_uri=model_uri, name=registry_name)
            print(f"Registered model version: {model_details.version}")


if __name__ == '__main__':
    main()