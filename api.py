from http.client import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd


def load_model(league):
   rf_model=joblib.load(f'Models/RFClassifier_{league}.pkl')## Premier League
   XGBoost_model=joblib.load(f'Models/XGBClassifier_{league}.pkl')## Premier League
   performances=joblib.load(f'Performances/Team Performances {league}.pkl')
   performances=pd.DataFrame(performances).reset_index(drop=True)
   team_profiles=joblib.load(f'Models/Clubs Data_{league}.pkl')

   return rf_model,XGBoost_model,performances,team_profiles

app=FastAPI(title='Match Predictions',description='This app predicts the results of the premier league'
                                                           ' according to RF Classifier and XGBoost Classifier')


def draw_filter(model):
  DRAW_THRESHOLD=0.27
  DRAW_PENALTY=0.6
  HOME_AWAY_BONUS=0.2

  filtered=model.copy()

  for i in range(len(filtered)):
    p_draw=filtered[i][1]

    if p_draw>=DRAW_THRESHOLD:
      filtered[i][1]=p_draw*DRAW_PENALTY
      filtered[i][0]+=p_draw*HOME_AWAY_BONUS
      filtered[i][2]+=p_draw*HOME_AWAY_BONUS
  return filtered

class MatchFeatures(BaseModel):
    league:str
    home_team:str
    away_team:str


   
@app.post("/predict")
def MakePrediction(match:MatchFeatures):

    rf_model,XGBoost_model,performances,team_profiles=load_model(match.league)
    
    if match.home_team not in team_profiles or match.away_team not in team_profiles:
        raise HTTPException(status_code=404,detail="Team Not Found")
    

    home=team_profiles[match.home_team]
    away=team_profiles[match.away_team]
    elo_diff=home['ELO Rating']-away['ELO Rating']
    points_diff=home['Total_Points']-away['Total_Points']
    features=[
        home['Days_Rest'],
        away['Days_Rest'],
        home['Avg_Scored_Last_5'],
        home['Avg_Conceded_Last_5'],
        home['Avg_Shots_Last_5'],
        home['Avg_Shots_Conceded_Last_5'],
        home['Wins_Last_5'],
        home['Losses_Last_5'],
        home['Total_Points'],
        home['Points_Last_5'],
        away['Avg_Scored_Last_5'],
        away['Avg_Conceded_Last_5'],
        away['Avg_Shots_Last_5'],
        away['Avg_Shots_Conceded_Last_5'],
        away['Wins_Last_5'],
        away['Losses_Last_5'],
        away['Total_Points'],
        away['Points_Last_5'],
        home['ELO Rating'],
        away['ELO Rating'],
        elo_diff,
        points_diff
        ]
    input_data=np.array([features])
    y_pred_rf=rf_model.predict_proba(input_data)
    y_pred_XG=XGBoost_model.predict_proba(input_data)
    y_pred=(y_pred_rf+y_pred_XG)/2
    y_pred=draw_filter(y_pred)

        
    home_df=performances[performances['Team']==match.home_team]
    away_df=performances[performances['Team']==match.away_team]

    home_performance=home_df.to_dict(orient='records')[0]
    away_performance=away_df.to_dict(orient='records')[0]


    final_pred=int(np.argmax(y_pred,axis=1)[0])
    points_map={0:f"{match.away_team}",1:"Draw",2:f"{match.home_team}"}
    return {
        "Match": f"{match.home_team} vs {match.away_team}",
        "Prediction": points_map[final_pred],
        "Home_Win_Prob": f"{y_pred[0][2] * 100:.1f}%",
        "Draw_Prob": f"{y_pred[0][1] * 100:.1f}%",
        "Away_Win_Prob": f"{y_pred[0][0] * 100:.1f}%",
        "Home Performance":home_performance,
        "Away Performance":away_performance
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)