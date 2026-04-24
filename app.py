import streamlit as st
from dotenv import load_dotenv
import os
import requests
import pandas as pd
import joblib

load_dotenv()
API_URL=os.getenv("API_URL")
if not API_URL:
    st.error("API not found")
st.title("Make a Prediction!")
st.write("You may wait for 50 seconds on your first prediction...")



TEAMS = {
    "Premier League": [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", 
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Burnley", 
        "Sunderland", "Liverpool", "Man City", "Man United", "Newcastle", 
        "Nott'm Forest", "Leeds", "Tottenham", "West Ham", "Wolves"
    ],
    "LaLiga": [
       'Real Madrid','Barcelona','Ath Madrid','Villarreal','Betis','Celta',
        'Sociedad','Getafe','Ath Bilbao','Osasuna','Espanol','Valencia',
        'Girona','Vallecano','Alaves','Sevilla',
        'Elche','Mallorca','Levante','Oviedo'
    ],
    "Greek Super League": [
        'Aris','Volos NFC','Olympiakos','AEK','PAOK',
        'Asteras Tripolis','Larisa','Panserraikos','Levadeiakos',
        'Kifisia','Atromitos','Panetolikos','Panathinaikos','OFI Crete'
    ],
    "Bundesliga": [
        'Bayern Munich','RB Leipzig','Ein Frankfurt',
          'Werder Bremen','Freiburg','Wolfsburg','Leverkusen','Hoffenheim',
          'Union Berlin','Stuttgard','St Pauli','Dortmund','Mainz',
          'FC Koln','M\'gladbach','Hamburg','Heidenheim','Augsburg'
    ]
}

def make_prediction(home_team,away_team,league):
    payload={
        "league": league,
        "home_team": home_team,
        "away_team": away_team
    }



    with st.spinner("Predicting..."):
        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                home_performance=data['Home Performance']
                away_performance=data['Away Performance']
                st.success(f"Winner Prediction: **{data['Prediction']}**")
                
                st.write("### Probabilities")
                home_prob = float(data['Home_Win_Prob'].replace('%', '')) / 100
                draw_prob = float(data['Draw_Prob'].replace('%', '')) / 100
                away_prob = float(data['Away_Win_Prob'].replace('%', '')) / 100
                
                st.progress(home_prob, text=f"{home_team}: {data['Home_Win_Prob']}")
                st.progress(draw_prob, text=f"Draw: {data['Draw_Prob']}")
                st.progress(away_prob, text=f"{away_team}: {data['Away_Win_Prob']}")

                st.divider()

                st.write(f"### {home_team} vs {away_team} Stats")
                home_data={
                "Scored":home_performance.get("Scored"),
                "Conceded":home_performance.get("Conceded"),
                "Total Shots":home_performance.get("Total Shots"),
                "Shots On Target":home_performance.get("Shots On Target"),
                "Wins":home_performance.get("Wins"),
                "Draws":home_performance.get("Draws"),
                "Losses":home_performance.get("Losses"),
                "Goal Difference":home_performance.get("Goal Difference"),
                "Goals Per Game":home_performance.get("Goals Per Game"),

                }

                away_data={
                    "Scored":away_performance.get("Scored"),
                    "Conceded":away_performance.get("Conceded"),
                    "Total Shots":away_performance.get("Total Shots"),
                    "Shots On Target":away_performance.get("Shots On Target"),
                    "Wins":away_performance.get("Wins"),
                    "Draws":away_performance.get("Draws"),
                    "Losses":away_performance.get("Losses"),
                    "Goal Difference":away_performance.get("Goal Difference"),
                    "Goals Per Game":away_performance.get("Goals Per Game"),
                }

                compared_data = pd.DataFrame({
                    home_team: home_data,
                    away_team: away_data
                })

                st.bar_chart(data=compared_data)
            elif response.status_code == 404:
                st.error("Club Not Found")
            else:
                st.error(f"API Error: Code {response.status_code}")
                
        except Exception as e:
            st.error("Something went wrong with the API")  

league = st.selectbox("Choose League",['Premier League','LaLiga','Bundesliga','Greek Super League'])

if 'home_team' not in st.session_state:
    st.session_state.home_team=TEAMS[league][0]
if 'away_team' not in st.session_state:
    st.session_state.away_team=TEAMS[league][1]

col1, col2, col3 = st.columns(3)

def swap_teams():
    st.session_state.home_team, st.session_state.away_team = st.session_state.away_team, st.session_state.home_team
with col1:
    st.selectbox("Home Team",TEAMS[league],
    index=None,placeholder="Select Away Team",key='home_team')
with col2:
    st.write(" ")
    st.write(" ")
    st.button("Switch",on_click=swap_teams,use_container_width=True)
        

with col3:
    st.selectbox("Away Team",TEAMS[league],
    index=None,placeholder="Select Away Team",key='away_team')

home_team=st.session_state.home_team
away_team=st.session_state.away_team


if home_team==away_team:
    st.warning("Please choose different teams")
elif st.button("Predict"):
    make_prediction(home_team,away_team,league)
    
