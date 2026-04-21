import streamlit as st
from dotenv import load_dotenv
import os
import requests
import pandas as pd

load_dotenv()
API_URL=os.getenv("API_URL")
if not API_URL:
    st.error("API not found")
st.title("Make a Prediction!")
st.write("You may wait for 50 seconds on your first prediction...")

st.text_input("Enter your name if you want!",key='name')
if 'name' in st.session_state:
    name=st.session_state.name
else:
    name='Anonymous'

st.write(f"### Welcome **{name}** !")

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



possible_answers=['Instagram','LinkedIn','Friend','Randomly','Prefer Not To Say']

with st.expander("Feedback Survey"):
    with st.form("Feedback Form "):

        st.slider('How satisfied were you with this predictions? (Your answer could help us improve our models)',
          min_value=0,
          max_value=10,
          step=1,
          key='Satisfaction')
        if 'Satisfaction' in st.session_state:
            satisfaction=st.session_state.Satisfaction
        else:
            satisfaction='Did Not Reply'

        source= st.selectbox('How did you find this page?',possible_answers)

        st.text_input('Do you have any comments?',key='comments')

        if 'comments' in st.session_state:
            comments=st.session_state.comments
        else:
            comments=' '

        submitted=st.form_submit_button("Submit")

        if submitted:
            st.success(f"Thank you {name} for your feedback. Appreciate it!")
            data={
                'Name':[name],
                'HomeTeam':[home_team],
                'AwayTeam':[away_team],
                'Satisfaction':[satisfaction],
                'Source':[source],
                'Comments':[comments]
            }

            file_path = os.path.join(current_dir, 'data.csv')
            if not os.path.isfile('data.csv'):
                data.to_csv('data.csv',index=False)
            else:
                data.to_csv('data.csv',mode='a',header=False,index=False)
            
