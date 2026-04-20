from http.client import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


rf_model=joblib.load('Models/RFClassifier_PL.pkl')## Premier League
XGBoost_model=joblib.load('Models/XGBClassifier_PL.pkl')## Premier League

rf_model_laliga=joblib.load('Models/RFClassifier_LaLiga.pkl')
XGBoost_model_laliga=joblib.load('Models/XGBClassifier_LaLiga.pkl')

rf_model_superleague=joblib.load('Models/RFClassifier_SuperLeague.pkl')
XGBoost_model_superleague=joblib.load('Models/XGBClassifier_SuperLeague.pkl')

rf_model_bundesliga=joblib.load('Models/RFClassifier_Bundesliga.pkl')
XGBoost_model_bundesliga=joblib.load('Models/XGBClassifier_Bundesliga.pkl')


app=FastAPI(title='Premier League Predictions',description='This app predicts the results of the premier league'
                                                           ' according to RF Classifier and XGBoost Classifier')
try:
    team_profiles=joblib.load('Models/Clubs Data_PL.pkl')
    team_profiles_laliga=joblib.load('Models/Clubs Data_LaLiga.pkl')
    team_profiles_superleague=joblib.load('Models/Clubs Data_Greece.pkl')
    team_profiles_bundesliga=joblib.load('Models/Clubs Data_Bundesliga.pkl')
except Exception as e:
    print("File of Club Data was not found ")

def draw_filter(model):
  DRAW_THRESHOLD=0.23
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
    
    if match.league.lower()=='laliga':
       if match.home_team not in team_profiles_laliga or match.away_team not in team_profiles_laliga:
        raise HTTPException(status_code=404,detail="Team Not Found")
    elif match.league.lower()=='premier league':
       if match.home_team not in team_profiles or match.away_team not in team_profiles:
          raise HTTPException(status_code=404,detail="Team Not Found")
    elif match.league.lower()=='greek super league':
       if match.home_team not in team_profiles_superleague or match.away_team not in team_profiles_superleague:
          raise HTTPException(status_code=404,detail="Team Not Found")
    elif match.league.lower()=='bundesliga':
        if match.home_team not in team_profiles_bundesliga or match.away_team not in team_profiles_bundesliga:
            raise HTTPException(status_code=404,detail="Team Not Found")
    else:
       raise HTTPException(status_code=400, detail="League not supported")
    

    if match.league.lower()=='premier league':
        home=team_profiles[match.home_team]
        away=team_profiles[match.away_team]
        features=[
          home['Days_Rest'],
          away['Days_Rest'],
          home['Avg_Scored_Last_5'],
          home['Avg_Conceded_Last_5'],
          home['Avg_Shots_Last_5'],
          home['Avg_Shots_Conceded_Last_5'],
          home['Wins_Last_5'],
          home['Losses_Last_5'],
          away['Avg_Scored_Last_5'],
          away['Avg_Conceded_Last_5'],
          away['Avg_Shots_Last_5'],
          away['Avg_Shots_Conceded_Last_5'],
          away['Wins_Last_5'],
          away['Losses_Last_5'],
          home['ELO Rating'],
          away['ELO Rating']
        ]
        input_data=np.array([features])
        y_pred_rf=rf_model.predict_proba(input_data)
        y_pred_XG=XGBoost_model.predict_proba(input_data)
        y_pred_pl=(y_pred_rf+y_pred_XG)/2
        y_pred=draw_filter(y_pred_pl)
    elif match.league.lower()=='laliga':
        home=team_profiles_laliga[match.home_team]
        away=team_profiles_laliga[match.away_team]
        features=[
          home['Days_Rest'],
          away['Days_Rest'],
          home['Avg_Scored_Last_5'],
          home['Avg_Conceded_Last_5'],
          home['Avg_Shots_Last_5'],
          home['Avg_Shots_Conceded_Last_5'],
          home['Wins_Last_5'],
          home['Losses_Last_5'],
          away['Avg_Scored_Last_5'],
          away['Avg_Conceded_Last_5'],
          away['Avg_Shots_Last_5'],
          away['Avg_Shots_Conceded_Last_5'],
          away['Wins_Last_5'],
          away['Losses_Last_5'],
          home['ELO Rating'],
          away['ELO Rating']
        ]

        input_data=np.array([features])
        y_pred_rf_laliga=rf_model_laliga.predict_proba(input_data)
        y_pred_XG_laliga=XGBoost_model_laliga.predict_proba(input_data)
        y_pred_laliga=(y_pred_rf_laliga*0.6+y_pred_XG_laliga*0.4)
        y_pred=draw_filter(y_pred_laliga)
    elif match.league.lower()=='greek super league':
       home=team_profiles_superleague[match.home_team]
       away=team_profiles_superleague[match.away_team]

       features=[
          home['Days_Rest'],
          away['Days_Rest'],
          home['Avg_Scored_Last_5'],
          home['Avg_Conceded_Last_5'],
          home['Avg_Shots_Last_5'],
          home['Avg_Shots_Conceded_Last_5'],
          home['Wins_Last_5'],
          home['Losses_Last_5'],
          away['Avg_Scored_Last_5'],
          away['Avg_Conceded_Last_5'],
          away['Avg_Shots_Last_5'],
          away['Avg_Shots_Conceded_Last_5'],
          away['Wins_Last_5'],
          away['Losses_Last_5'],
          home['ELO Rating'],
          away['ELO Rating']
        ]
       input_data=np.array([features])
       y_pred_rf_superleague=rf_model_superleague.predict_proba(input_data)
       y_pred_XG_superleague=XGBoost_model_superleague.predict_proba(input_data)
       y_pred_superleague=(y_pred_rf_superleague*0.6+y_pred_XG_superleague*0.4)
       y_pred=draw_filter(y_pred_superleague)
    elif match.league.lower()=='bundesliga':
       home=team_profiles_bundesliga[match.home_team]
       away=team_profiles_bundesliga[match.away_team]
       features=[
          home['Days_Rest'],
          away['Days_Rest'],
          home['Avg_Scored_Last_5'],
          home['Avg_Conceded_Last_5'],
          home['Avg_Shots_Last_5'],
          home['Avg_Shots_Conceded_Last_5'],
          home['Wins_Last_5'],
          home['Losses_Last_5'],
          away['Avg_Scored_Last_5'],
          away['Avg_Conceded_Last_5'],
          away['Avg_Shots_Last_5'],
          away['Avg_Shots_Conceded_Last_5'],
          away['Wins_Last_5'],
          away['Losses_Last_5'],
          home['ELO Rating'],
          away['ELO Rating']
        ]
       input_data=np.array([features])
       y_pred_rf_bundesliga=rf_model_bundesliga.predict_proba(input_data)
       y_pred_XG_bundesliga=XGBoost_model_bundesliga.predict_proba(input_data)
       y_pred=(y_pred_rf_bundesliga*0.6+y_pred_XG_bundesliga*0.4)
       y_pred=draw_filter(y_pred)

    final_pred=int(np.argmax(y_pred,axis=1)[0])
    points_map={0:f"{match.away_team}",1:"Draw",2:f"{match.home_team}"}
    return {
        "Match": f"{match.home_team} vs {match.away_team}",
        "Prediction": points_map[final_pred],
        "Home_Win_Prob": f"{y_pred[0][2] * 100:.1f}%",
        "Draw_Prob": f"{y_pred[0][1] * 100:.1f}%",
        "Away_Win_Prob": f"{y_pred[0][0] * 100:.1f}%"
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)