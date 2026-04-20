import streamlit as st
from dotenv import load_dotenv
import os
import requests

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

league = st.selectbox("Choose League",['Premier League','LaLiga','Bundesliga','Greek Super League'],accept_new_options=False)
col1, col2 = st.columns(2)

with col1:
    home_team=st.selectbox("Home Team",TEAMS[league],accept_new_options=False)
with col2:
    away_team=st.selectbox("Away Team",TEAMS[league],accept_new_options=False)
if home_team==away_team:
    st.warning("Please choose different teams")
elif st.button("Predict"):
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
                st.success(f"Winner Prediction: **{data['Prediction']}**")
                
                st.write("### Probabilities")
                home_prob = float(data['Home_Win_Prob'].replace('%', '')) / 100
                draw_prob = float(data['Draw_Prob'].replace('%', '')) / 100
                away_prob = float(data['Away_Win_Prob'].replace('%', '')) / 100
                
                st.progress(home_prob, text=f"{home_team}: {data['Home_Win_Prob']}")
                st.progress(draw_prob, text=f"Draw: {data['Draw_Prob']}")
                st.progress(away_prob, text=f"{away_team}: {data['Away_Win_Prob']}")
            
            elif response.status_code == 404:
                st.error("Club Not Found")
            else:
                st.error(f"API Error: Code {response.status_code}")
                
        except Exception as e:
            st.error("The API is not responding right now. Please try again later.")