import Preprocessing
import rf_classifier
import xgb_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib
import sys
import os

def dataset(path,league):
    if league=='Premier League':
        dataset=pd.read_csv(path).iloc[:,:24]
    else:
        dataset=pd.read_csv(path).iloc[:,:23]
    dataset,team_stats,teams_elo=Preprocessing.preprocess(dataset,league)
    dataset['ELO_Diff']=dataset['Home_ELO_Score']-dataset['Away_ELO_Score']

    dataset=dataset.sort_values('Date')
    dataset=dataset.drop(columns=['Date','FTAG','FTHG','HTR','HS',
                                        'AS','Day','Month','HY','AY',
                                        'HTHG','HTAG','HST','AST',
                                        'HF','AF','HC','AC'])
    X=dataset.drop(columns=['FTR']).values
    y=dataset['FTR'].values


    return dataset,X,y,team_stats,teams_elo

leagues_config = {
    'Premier League':'datasets/E0 (1).csv',
    'LaLiga':'datasets/SP1.csv',
    'Bundesliga':'datasets/D1.csv',
    'Greek Super League':'datasets/G1.csv'
}


os.makedirs('Models', exist_ok=True)

for league_name, file_path in leagues_config.items():
    print("="*30)
    print(f"{league_name} model creation just started".upper())
    df,X,y,team_stats,teams_elo=dataset(file_path, league_name)
    
    le=LabelEncoder()
    y_encoded=le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, shuffle=False
    )
    
    X_train = X_train[:, 2:]
    X_test = X_test[:, 2:]
   

    rf_model = rf_classifier.rf_model(X_train, y_train)
    y_pred_rf = rf_classifier.make_prediction(rf_model, X_test)
    
    xgb_model=xgb_classifier.xgb_model(X_train, y_train)
    y_pred_xgb=xgb_classifier.make_prediction(xgb_model, X_test)
    
    
    if league_name=='Premier League':
        joblib.dump(rf_model,'Models/RFClassifier_PL.pkl')
        print(f"RF Model of {league_name} saved")
        joblib.dump(xgb_model,'Models/XGBClassifier_PL.pkl')
        print(f"XGB Model of {league_name} saved")
    elif league_name=='Greek Super League':
        joblib.dump(rf_model,'Models/RFClassifier_SuperLeague.pkl')
        print(f"RF Model of {league_name} saved")
        joblib.dump(xgb_model,'Models/XGBClassifier_SuperLeague.pkl')
        print(f"XGB Model of {league_name} saved")
    else:
        joblib.dump(rf_model,f'Models/RFClassifier_{league_name}.pkl')
        print(f"RF Model of {league_name} saved")
        joblib.dump(xgb_model,f'Models/XGBClassifier_{league_name}.pkl')
        print(f"XGB Model of {league_name} saved")
    
    
    all_teams=df['HomeTeam'].unique()
    current_team_data={}

    for team in all_teams:
        team_data=team_stats[team_stats['Team']==team].iloc[-1]

        current_team_data[team]={
            'Team':team,
            'ELO Rating':teams_elo.get(team),
            'Days_Rest':team_data['Days_Rest'],
            'Avg_Scored_Last_5':team_data['Avg_Scored_Last_5'],
            'Avg_Conceded_Last_5':team_data['Avg_Conceded_Last_5'],
            'Avg_Shots_Last_5':team_data['Avg_Shots_Last_5'],
            'Avg_Shots_Conceded_Last_5':team_data['Avg_Shots_Conceded_Last_5'],
            'Wins_Last_5':team_data['Wins_Last_5'],
            'Losses_Last_5':team_data['Losses_Last_5']
        }
    if league_name=='Premier League':
        joblib.dump(current_team_data,'Models/Clubs Data_PL.pkl')
        print(f"Club Data of {league_name} saved")
    elif league_name=='Greek Super League':
        joblib.dump(current_team_data,'Models/Clubs Data_Greece.pkl')
        print(f"Club Data of {league_name} saved")
    else:
        joblib.dump(current_team_data,f'Models/Clubs Data_{league_name}.pkl')
        print(f"Club Data of {league_name} saved")
    
        

