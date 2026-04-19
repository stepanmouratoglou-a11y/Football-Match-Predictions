import streamlit as st
import requests

api_url="https://football-predictions-api-ope0.onrender.com/docs#/default/MakePrediction_predict_post"
st.title("Predict the outcome of your favourite club's next game!")


TEAMS = {
    "Premier League": [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", 
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich", 
        "Leicester", "Liverpool", "Man City", "Man United", "Newcastle", 
        "Nott'm Forest", "Southampton", "Tottenham", "West Ham", "Wolves"
    ],
    "LaLiga": [
        "Alaves", "Ath Bilbao", "Ath Madrid", "Barcelona", "Betis", 
        "Celta", "Espanol", "Getafe", "Girona", "Las Palmas", 
        "Leganes", "Mallorca", "Osasuna", "Real Madrid", "Sociedad", 
        "Sevilla", "Valencia", "Valladolid", "Vallecano", "Villarreal"
    ],
    "Greek Super League": [
        "AEK", "Aris", "Asteras Tripolis", "Atromitos", "Athens Kallithea", 
        "Lamia", "Levadiakos", "OFI", "Olympiakos", "Panathinaikos", 
        "Panaitolikos", "Panserraikos", "PAOK", "Volos NFC"
    ],
    "Bundesliga": [
        "Augsburg", "Bochum", "Bremen", "Dortmund", "Ein Frankfurt", 
        "Freiburg", "Heidenheim", "Hoffenheim", "Holstein Kiel", "Leipzig", 
        "Leverkusen", "Mainz", "M'gladbach", "Bayern Munich", "St Pauli", 
        "Stuttgart", "Union Berlin", "Wolfsburg"
    ]
}

league = st.selectbox("Choose League", list(TEAMS.keys()))
col1, col2 = st.columns(2)

with col1:
    home_team=st.selectbox("Home Team",TEAMS[league])
    home_rest=7
with col2:
    away_team=st.selectbox("Away Team",TEAMS[league])
    away_rest=7
if home_team==away_team:
    st.warning("Please choose different teams")
elif st.button("Predict"):
    payload={
        "league": league,
        "home_team": home_team,
        "away_team": away_team,
        "home_days_rest": home_rest,
        "away_days_rest": away_rest
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
                
                st.progress(home_prob, text=f"Home ({home_team}): {data['Home_Win_Prob']}")
                st.progress(draw_prob, text=f"Χ (Draw): {data['Draw_Prob']}")
                st.progress(away_prob, text=f"Away ({away_team}): {data['Away_Win_Prob']}")
            
            elif response.status_code == 404:
                st.error("Club Not Found")
            else:
                st.error(f"API Error: Code {response.status_code}")
                
        except Exception as e:
            st.error("The API is not responding right now. Please try again later.")