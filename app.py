import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import colorsys
import hashlib
import altair as alt

st.set_page_config(
    page_title="Football Match Predictions & Stats",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

LEAGUE_LOGO_MAPS = {
    "Premier League": {
        "Arsenal": "Images/Clubs/Arsenal.png",
        "Aston Villa": "Images/Clubs/Aston_Villa.png",
        "Bournemouth": "Images/Clubs/Bournemouth.webp",
        "Brentford": "Images/Clubs/Brentford_FC_crest.svg",
        "Brighton": "Images/Clubs/Brighton.webp",
        "Burnley": "Images/Clubs/Burnley.webp",
        "Chelsea": "Images/Clubs/Chelsea.png",
        "Crystal Palace": "Images/Clubs/Crystal_Palace.png",
        "Everton": "Images/Clubs/Everton.png",
        "Fulham": "Images/Clubs/Fulham.png",
        "Leeds": "Images/Clubs/Leeds_United.svg",
        "Liverpool": "Images/Clubs/Liverpool.png",
        "Man City": "Images/Clubs/Manchester_City.svg",
        "Man United": "Images/Clubs/Manchester_United.svg",
        "Newcastle": "Images/Clubs/Newcastle.webp",
        "Nott'm Forest": "Images/Clubs/Nottingham_Forest.png",
        "Sunderland": "Images/Clubs/Sunderland.png",
        "Tottenham": "Images/Clubs/Tottenham.png",
        "West Ham": "Images/Clubs/West_Ham.png",
        "Wolves": "Images/Clubs/Wolves.webp"
    },
    "Bundesliga": {
        "Augsburg": "Images/Bundesliga/FC_Augsburg_logo.svg",
        "Bayern Munich": "Images/Bundesliga/Bayern.png",
        "Dortmund": "Images/Bundesliga/Borussia_Dortmund_logo.svg.webp",
        "FC Koln": "Images/Bundesliga/FC_Koln.png",
        "Hamburg": "Images/Bundesliga/Hamburg.png",
        "RB Leipzig": "Images/Bundesliga/RB_Leipzig.webp",
        "Leverkusen": "Images/Bundesliga/bayer_leverkusen.png",
        "Werder Bremen": "Images/Bundesliga/bremen.png",
        "Ein Frankfurt": "Images/Bundesliga/frankfurt.png",
        "Freiburg": "Images/Bundesliga/freiburg.png",
        "M'gladbach": "Images/Bundesliga/gladbach.png",
        "Heidenheim": "Images/Bundesliga/heideheim.png",
        "Hoffenheim": "Images/Bundesliga/hoffenheim.png",
        "St Pauli": "Images/Bundesliga/st_pauli.png",
        "Stuttgart": "Images/Bundesliga/stuttgard.png",
        "Union Berlin": "Images/Bundesliga/union_berlin.png",
        "Wolfsburg": "Images/Bundesliga/wolfsburg.png"
    },
    "LaLiga": {
        "Alaves": "Images/LaLiga/alaves.png",
        "Ath Bilbao": "Images/LaLiga/athletic_club_bilbao.png",
        "Ath Madrid": "Images/LaLiga/atletico_madrid.png",
        "Barcelona": "Images/LaLiga/Barcelona.png",
        "Betis": "Images/LaLiga/real_betis.png",
        "Celta": "Images/LaLiga/celta.png",
        "Elche": "Images/LaLiga/elche.png",
        "Espanol": "Images/LaLiga/espanyol.png",
        "Getafe": "Images/LaLiga/getafe.jpg",
        "Girona": "Images/LaLiga/girona.png",
        "Levante": "Images/LaLiga/levante.png",
        "Mallorca": "Images/LaLiga/mallorca.png",
        "Osasuna": "Images/LaLiga/osasuna.png",
        "Oviedo": "Images/LaLiga/real_oviedo.png",
        "Real Madrid": "Images/LaLiga/real_madrid.png",
        "Sevilla": "Images/LaLiga/sevilla.png",
        "Sociedad": "Images/LaLiga/real_sociedad.png",
        "Valencia": "Images/LaLiga/valencia.png",
        "Vallecano": "Images/LaLiga/rayo_vallecano.png",
        "Villarreal": "Images/LaLiga/villareal.png"
    },
    "Greek Super League": {
        "AEK": "Images/GreekSuperLeague/AEK.png",
        "Aris": "Images/GreekSuperLeague/Aris.png",
        "Asteras Tripolis": "Images/GreekSuperLeague/Asteras_tripolis_aktor.png",
        "Atromitos": "Images/GreekSuperLeague/Atromitos.png",
        "Kifisia": "Images/GreekSuperLeague/Kifisia.png",
        "Larisa": "Images/GreekSuperLeague/Larissa.jpg",
        "Levadeiakos": "Images/GreekSuperLeague/levadeiakos.png",
        "OFI Crete": "Images/GreekSuperLeague/OFI.jpg",
        "Olympiakos": "Images/GreekSuperLeague/Olympiacos.png",
        "PAOK": "Images/GreekSuperLeague/PAOK.jpg",
        "Panathinaikos": "Images/GreekSuperLeague/Panathinaikos.png",
        "Panetolikos": "Images/GreekSuperLeague/Panetolikos.png",
        "Panserraikos": "Images/GreekSuperLeague/Panserraikos.png",
        "Volos NFC": "Images/GreekSuperLeague/Volos.png"
    }
}

TEAM_COLOR_MAP = {
    "Arsenal": "#EF0107",       # Red
    "Aston Villa": "#941C44",   # Claret
    "Bournemouth": "#D71920",   # Red
    "Brentford": "#E30613",     # Red
    "Brighton": "#0057B8",      # Blue
    "Burnley": "#6C1D45",       # Claret
    "Chelsea": "#034694",       # Royal Blue
    "Crystal Palace": "#1B458F",# Blue
    "Everton": "#003399",       # Royal Blue
    "Fulham": "#000000",        # Black
    "Leeds": "#1D428A",         # Blue
    "Liverpool": "#C8102E",     # Red
    "Man City": "#6CABDD",      # Sky Blue
    "Man United": "#DA291C",    # Red
    "Newcastle": "#241F20",     # Black
    "Nott'm Forest": "#DD0000", # Red
    "Sunderland": "#FF0000",    # Red
    "Tottenham": "#132257",     # Navy Blue
    "West Ham": "#7C2C3B",      # Claret
    "Wolves": "#FDB913"         # Gold
}

def get_team_color(team_name):
    """Returns the primary hex color of a team based on its brand/logo, or generates a deterministic one."""
    if team_name in TEAM_COLOR_MAP:
        return TEAM_COLOR_MAP[team_name]
    
    # Generate deterministic custom color for other league clubs
    hash_object = hashlib.md5(team_name.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    
    hue = (int(hash_hex[:4], 16) % 360) / 360.0
    sat = 0.6 + (int(hash_hex[4:8], 16) % 20) / 100.0   # Saturation: 60% to 80%
    val = 0.45 + (int(hash_hex[8:12], 16) % 15) / 100.0  # Lightness: 45% to 60%
    
    rgb = colorsys.hls_to_rgb(hue, val, sat)
    hex_color = '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255), 
        int(rgb[1] * 255), 
        int(rgb[2] * 255)
    )
    return hex_color

def get_team_logo_path(team_name, league_name):
    """Returns absolute/relative path to a team's logo if it exists."""
    league_map = LEAGUE_LOGO_MAPS.get(league_name, {})
    path = league_map.get(team_name)
    if path and os.path.exists(path):
        return path
    return None

@st.cache_resource
def load_models_and_data(league_name):
    """Loads classification models, ELO/stats profiles, and performances for a league."""
    rf_path = f'Models/RFClassifier_{league_name}.pkl'
    xgb_path = f'Models/XGBClassifier_{league_name}.pkl'
    clubs_path = f'Models/Clubs Data_{league_name}.pkl'
    perf_path = f'Performances/Team Performances {league_name}.pkl'
    
    rf_model = joblib.load(rf_path)
    xgb_model = joblib.load(xgb_path)
    team_profiles = joblib.load(clubs_path)
    performances = joblib.load(perf_path)
    
    performances = pd.DataFrame(performances).reset_index(drop=True)
    return rf_model, xgb_model, team_profiles, performances

def draw_filter(y_prob):
    """Applies a draw penalty filter matching the model training/api specification."""
    DRAW_THRESHOLD = 0.3
    DRAW_PENALTY = 0.7
    HOME_AWAY_BONUS = 0.15

    filtered = y_prob.copy()
    for i in range(len(filtered)):
        p_draw = filtered[i][1]
        if p_draw >= DRAW_THRESHOLD:
            filtered[i][1] = p_draw * DRAW_PENALTY
            filtered[i][0] += p_draw * HOME_AWAY_BONUS
            filtered[i][2] += p_draw * HOME_AWAY_BONUS
    return filtered

def make_prediction_ensemble(home_team, away_team, rf_model, xgb_model, team_profiles):
    """Runs prediction locally using the ensemble of RF and XGBoost classifiers."""
    home = team_profiles[home_team]
    away = team_profiles[away_team]
    
    elo_diff = home['ELO Rating'] - away['ELO Rating']
    points_diff = home['Total_Points'] - away['Total_Points']
    
    features = [
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
    
    feature_names = [
        'Home_Days_Rest', 'Away_Days_Rest',
        'Home_Avg_Scored_5', 'Home_Avg_Conceded_5', 'Home_Avg_Shots_Last_5', 'Home_Avg_Shots_Conceded_Last_5',
        'Home_Wins_Last_5', 'Home_Losses_Last_5', 'Home_Total_Points', 'Home_Points_Last_5',
        'Away_Avg_Scored_5', 'Away_Avg_Conceded_5', 'Away_Avg_Shots_Last_5', 'Away_Avg_Shots_Conceded_Last_5',
        'Away_Wins_Last_5', 'Away_Losses_Last_5', 'Away_Total_Points', 'Away_Points_Last_5',
        'Home_ELO_Score', 'Away_ELO_Score', 'ELO_Diff', 'Points_Diff'
    ]
    
    input_df = pd.DataFrame([features], columns=feature_names)
    
    y_pred_rf = rf_model.predict_proba(input_df)
    y_pred_xgb = xgb_model.predict_proba(input_df)
    
    y_pred = (y_pred_rf + y_pred_xgb) / 2
    y_pred = draw_filter(y_pred)
    
    final_pred_idx = int(np.argmax(y_pred, axis=1)[0])
    
    points_map = {0: away_team, 1: "Draw", 2: home_team}
    
    return {
        "Prediction": points_map[final_pred_idx],
        "Home_Win_Prob": float(y_pred[0][2]),
        "Draw_Prob": float(y_pred[0][1]),
        "Away_Win_Prob": float(y_pred[0][0])
    }

st.sidebar.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <h2 style='color: #3D195A; font-weight: bold;'>⚽ Football Predictor</h2>
    </div>
    """, 
    unsafe_allow_html=True
)

league = st.sidebar.selectbox(
    "Select Football League", 
    ['Premier League', 'LaLiga', 'Bundesliga', 'Greek Super League'],
    index=0
)

# Dynamic League Logo
LEAGUE_LOGO_MAP = {
    'Premier League': 'Images/PremierLeague_Logo/PremierLeague_Logo.png',
    'LaLiga': 'Images/LaLiga/LaLiga_logo.webp',
    'Bundesliga': 'Images/Bundesliga/Bundesliga_Logo.jpg',
    'Greek Super League': 'Images/GreekSuperLeague/Greek_Super_League_Logo.jpg'
}

league_logo_path = LEAGUE_LOGO_MAP.get(league)
if league_logo_path and os.path.exists(league_logo_path):
    st.sidebar.image(league_logo_path, use_container_width=True)

st.sidebar.divider()

page = st.sidebar.radio(
    "Select Navigation Page", 
    ["📊 2025/2026 Season Performance", "🔮 Match Predictor"],
    index=0
)

try:
    rf_model, xgb_model, team_profiles, performances = load_models_and_data(league)
except Exception as e:
    st.sidebar.error(f"Error loading models/data for {league}: {e}")
    st.stop()

teams_list = sorted(list(performances['Team'].unique()))

df_table = performances.copy()
df_table['Played'] = df_table['Wins'] + df_table['Draws'] + df_table['Losses']

df_table = df_table.sort_values(
    by=['Points', 'Goal Difference', 'Scored'], 
    ascending=[False, False, False]
).reset_index(drop=True)

df_table.index += 1  
df_table.index.name = 'Pos'
df_table_display = df_table.reset_index()

if page == "📊 2025/2026 Season Performance":
    st.markdown(f"# 📊 {league} — 2025/2026 Season Performance")
    st.markdown("Detailed club statistics, interactive comparisons, and final standings.")
    st.divider()

    # --- EXTENDED LEAGUE-WIDE SEASON OVERVIEW ---
    st.subheader("📊 Season Overview & Insights")
    
    total_played = int(df_table['Played'].sum() / 2)
    total_goals = int(df_table['Scored'].sum())
    avg_goals = total_goals / total_played if total_played > 0 else 0
    
    top_scorer_row = df_table.loc[df_table['Scored'].idxmax()]
    best_defense_row = df_table.loc[df_table['Conceded'].idxmin()]
    most_discipline_row = df_table.loc[df_table['Yellow Cards'].idxmin()]
    
    win_pct = (df_table['Wins'].sum() / total_played) * 100 if total_played > 0 else 0
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Matches", total_played)
    c2.metric("Total Goals", f"{total_goals} ({avg_goals:.2f}/match)")
    c3.metric("Decisive Matches", f"{win_pct:.1f}%", help="Percentage of matches ending in a win/loss")
    
    c4, c5, c6 = st.columns(3)
    c4.metric("Most Goals", f"{top_scorer_row['Team']}", f"{int(top_scorer_row['Scored'])} Goals")
    c5.metric("Best Defence", f"{best_defense_row['Team']}", f"{int(best_defense_row['Conceded'])} Conceded")
    c6.metric("Cleanest Team", f"{most_discipline_row['Team']}", f"{int(most_discipline_row['Yellow Cards'])} 🟨 Cards")
    
    st.divider()

    st.subheader("🏆 Final Standings")
    table_cols = ['Pos', 'Team', 'Played', 'Wins', 'Draws', 'Losses', 'Scored', 'Conceded', 'Goal Difference', 'Points']
    
    st.dataframe(
        df_table_display[table_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Pos": st.column_config.NumberColumn("Position", format="%d"),
            "Team": st.column_config.TextColumn("Club"),
            "Played": st.column_config.NumberColumn("P"),
            "Wins": st.column_config.NumberColumn("W"),
            "Draws": st.column_config.NumberColumn("D"),
            "Losses": st.column_config.NumberColumn("L"),
            "Scored": st.column_config.NumberColumn("GF"),
            "Conceded": st.column_config.NumberColumn("GA"),
            "Goal Difference": st.column_config.NumberColumn("GD"),
            "Points": st.column_config.NumberColumn("Pts")
        }
    )
    
    st.divider()

    st.subheader("🎯 Interactive Dashboards")
    tab1, tab2, tab3 = st.tabs([
        "🛡️ Club Performance Explorer", 
        "📈 League Metric Comparison", 
        "⚔️ Compare Clubs Side-by-Side"
    ])

    with tab1: 
        st.markdown("### Club Profile Overview")
        col_select, col_empty = st.columns([2, 3])
        with col_select:
            selected_club = st.selectbox("Select a Club", teams_list, key="single_club_select")

        club_row = df_table[df_table['Team'] == selected_club].iloc[0]
        pos = df_table[df_table['Team'] == selected_club].index[0]
        club_logo = get_team_logo_path(selected_club, league)

        c1, c2 = st.columns([1, 4])
        with c1:
            if club_logo:
                st.image(club_logo, use_container_width=True)
            else:
                st.markdown(
                    "<div style='font-size: 72px; text-align: center; line-height: 100px;'>⚽</div>", 
                    unsafe_allow_html=True
                )
        with c2:
            st.markdown(f"## {selected_club}")
            st.markdown(f"**Standings Position:** #{pos} | **League Points:** {club_row['Points']} pts")

        st.write("#### Performance Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Wins", int(club_row['Wins']))
        m2.metric("Draws", int(club_row['Draws']))
        m3.metric("Losses", int(club_row['Losses']))
        m4.metric("Goals For", int(club_row['Scored']))
        m5.metric("Goals Against", int(club_row['Conceded']))
        m6.metric("Goal Difference", int(club_row['Goal Difference']))

        st.write("#### Season Statistics Breakdown")
        stats_data = pd.DataFrame({
            "Stat Metric": [
                "Shots Made", "Shots Conceded", 
                "Shots On Target", "Shots On Target Conceded", 
                "Corners Won", "Corners Conceded",
                "Fouls Committed", "Fouls Suffered"
            ],
            "Count": [
                club_row['Total Shots'], club_row['Shots_Conceded'],
                club_row['Shots On Target'], club_row['Shots_ontarget_conceded'],
                club_row['Corners'], club_row['Corners_conceded'],
                club_row['Fouls_commited'], club_row['Fouls_suffered']
            ]
        })
        st.bar_chart(stats_data, x="Stat Metric", y="Count", color=get_team_color(selected_club))

    with tab2:
        st.markdown("### League-Wide Feature Comparison")
        FEATURE_MAP = {
            'Points': 'Points',
            'Scored': 'Goals Scored (GF)',
            'Conceded': 'Goals Conceded (GA)',
            'Goal Difference': 'Goal Difference (GD)',
            'Total Shots': 'Shots Made',
            'Shots_Conceded': 'Shots Conceded',
            'Shots On Target': 'Shots on Target',
            'Corners': 'Corners Won',
            'Wins': 'Wins',
            'Losses': 'Losses',
            'Goals Per Game': 'Average Goals per Game',
            'Goals Conceded Per Game': 'Average Goals Conceded per Game',
            'Shots Per Game': 'Average Shots per Game'
        }
        
        c1, c2 = st.columns([2, 3])
        with c1:
            selected_feature_name = st.selectbox("Select a Feature", list(FEATURE_MAP.values()), key="feature_select")
            selected_feature = [k for k, v in FEATURE_MAP.items() if v == selected_feature_name][0]
            
        chart_data = performances[['Team', selected_feature]].copy()
        chart_data = chart_data.sort_values(by=selected_feature, ascending=False)
        
        chart_data['Highlight'] = chart_data['Team'].apply(
            lambda x: f"Selected ({selected_club})" if x == selected_club else "Others"
        )
        
        st.write(f"#### Comparative Chart: {selected_feature_name}")
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Team:N', sort='-y', title='Club'),
            y=alt.Y(f'{selected_feature}:Q', title=selected_feature_name),
            color=alt.Color('Highlight:N', scale=alt.Scale(
                domain=[f"Selected ({selected_club})", "Others"],
                range=[get_team_color(selected_club), "#D3D3D3"]
            ), title=None)
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

    with tab3:
        st.markdown("### Compare Performances of 1 or More Clubs")
        
        compare_clubs = st.multiselect(
            "Select Clubs to Compare", 
            teams_list, 
            default=[teams_list[0], teams_list[1]] if len(teams_list) > 1 else [teams_list[0]],
            key="compare_clubs_select"
        )
        
        if compare_clubs:
            comparison_df = df_table[df_table['Team'].isin(compare_clubs)].copy()
            
            stats_list = [
                'Points', 'Wins', 'Draws', 'Losses', 
                'Scored', 'Conceded', 'Goal Difference', 
                'Total Shots', 'Shots_Conceded', 
                'Shots On Target', 'Corners', 
                'Goals Per Game', 'Goals Conceded Per Game'
            ]
            
            comp_table = comparison_df.set_index('Team')[stats_list].T
            st.write("#### Detailed Statistical Comparison")
            st.dataframe(comp_table, use_container_width=True)
            
            st.write("#### Points, Goals For, and Goals Against Comparison")
            chart_compare = comparison_df[['Team', 'Points', 'Scored', 'Conceded']].melt(
                id_vars='Team', 
                var_name='Metric', 
                value_name='Value'
            )
            chart = alt.Chart(chart_compare).mark_bar().encode(
                x=alt.X('Metric:N', title=None),
                y=alt.Y('Value:Q', title='Value'),
                color=alt.Color('Team:N', scale=alt.Scale(
                    domain=list(compare_clubs),
                    range=[get_team_color(team) for team in compare_clubs]
                ), title='Club'),
                xOffset='Team:N'
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Please select at least one club to compare.")

elif page == "🔮 Match Predictor":
    st.markdown(f"# 🔮 {league} Match Predictor")
    st.markdown("Calculate the outcome probability of an upcoming fixture using local machine learning ensembles.")
    st.divider()

    if 'home_team' not in st.session_state or st.session_state.home_team not in teams_list:
        st.session_state.home_team = teams_list[0] if len(teams_list) > 0 else None
    if 'away_team' not in st.session_state or st.session_state.away_team not in teams_list:
        st.session_state.away_team = teams_list[1] if len(teams_list) > 1 else None

    def swap_teams():
        h = st.session_state.home_team
        a = st.session_state.away_team
        st.session_state.home_team = a
        st.session_state.away_team = h

    col1, col2, col3 = st.columns([5, 2, 5])
    
    with col1:
        st.markdown("### 🏠 Home Team")
        home_team = st.selectbox(
            "Select Home Team", 
            teams_list, 
            key='home_team'
        )
        home_logo = get_team_logo_path(home_team, league)
        if home_logo:
            st.image(home_logo, width=120)
        else:
            st.markdown("<div style='font-size: 64px;'>🏠</div>", unsafe_allow_html=True)
            
    with col2:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.button("🔄 Swap Teams", on_click=swap_teams, use_container_width=True)
        
    with col3:
        st.markdown("### ✈️ Away Team")
        away_team = st.selectbox(
            "Select Away Team", 
            teams_list, 
            key='away_team'
        )
        away_logo = get_team_logo_path(away_team, league)
        if away_logo:
            st.image(away_logo, width=120)
        else:
            st.markdown("<div style='font-size: 64px;'>✈️</div>", unsafe_allow_html=True)

    st.divider()

    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'prev_home_team' not in st.session_state:
        st.session_state.prev_home_team = None
    if 'prev_away_team' not in st.session_state:
        st.session_state.prev_away_team = None
    if 'prev_league' not in st.session_state:
        st.session_state.prev_league = None

    if (st.session_state.prev_home_team != home_team or 
        st.session_state.prev_away_team != away_team or 
        st.session_state.prev_league != league):
        st.session_state.prediction_results = None
        st.session_state.prev_home_team = home_team
        st.session_state.prev_away_team = away_team
        st.session_state.prev_league = league

    if home_team == away_team:
        st.warning("⚠️ Home and Away teams must be different. Please select distinct clubs.")
    else:
        predict_clicked = st.button("🔮 Calculate Match Probabilities", type="primary", use_container_width=True)
        
        if predict_clicked:
            with st.spinner("Analyzing team data and calculating predictions..."):
                try:
                    res = make_prediction_ensemble(home_team, away_team, rf_model, xgb_model, team_profiles)
                    st.session_state.prediction_results = res
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.session_state.prediction_results = None

        if st.session_state.prediction_results is not None:
            res = st.session_state.prediction_results
            winner = res["Prediction"]
            h_prob = res["Home_Win_Prob"]
            d_prob = res["Draw_Prob"]
            a_prob = res["Away_Win_Prob"]
            
            st.success(f"### Predicted Winner: **{winner}**")
            
            st.write("#### Match Probabilities breakdown:")
            st.progress(h_prob, text=f"🏠 {home_team} Win Probability: {h_prob * 100:.1f}%")
            st.progress(d_prob, text=f"🤝 Draw Probability: {d_prob * 100:.1f}%")
            st.progress(a_prob, text=f"✈️ {away_team} Win Probability: {a_prob * 100:.1f}%")
            
            st.divider()
            st.write(f"### Stats Comparison: {home_team} vs {away_team}")
            
            h_perf = performances[performances['Team'] == home_team].iloc[0]
            a_perf = performances[performances['Team'] == away_team].iloc[0]
            h_profile = team_profiles[home_team]
            a_profile = team_profiles[away_team]
            
            col_h, col_a = st.columns(2)
            with col_h:
                st.markdown(f"#### 🏠 {home_team} Season Overview")
                h_pos = df_table[df_table['Team'] == home_team].index[0]
                st.write(f"**Position:** #{h_pos} | **ELO Rating:** {int(h_profile['ELO Rating'])} | **Points:** {int(h_perf['Points'])} pts")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Wins", int(h_perf['Wins']))
                m2.metric("Draws", int(h_perf['Draws']))
                m3.metric("Losses", int(h_perf['Losses']))
                
                m4, m5, m6 = st.columns(3)
                m4.metric("Goals Scored", f"{int(h_perf['Scored'])} ({h_perf['Goals Per Game']:.2f}/g)")
                m5.metric("Goals Conceded", f"{int(h_perf['Conceded'])} ({h_perf['Goals Conceded Per Game']:.2f}/g)")
                m6.metric("Shots/Game", f"{h_perf['Shots Per Game']:.2f} ({h_perf['SoT Per Game']:.2f} SoT)")
                
                st.markdown("**Recent Form (Last 5 matches):**")
                rf1, rf2, rf3 = st.columns(3)
                rf1.metric("Form Points", int(h_profile['Points_Last_5']))
                rf2.metric("Form Wins", int(h_profile['Wins_Last_5']))
                rf3.metric("Form Losses", int(h_profile['Losses_Last_5']))
                
            with col_a:
                st.markdown(f"#### ✈️ {away_team} Season Overview")
                a_pos = df_table[df_table['Team'] == away_team].index[0]
                st.write(f"**Position:** #{a_pos} | **ELO Rating:** {int(a_profile['ELO Rating'])} | **Points:** {int(a_perf['Points'])} pts")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Wins", int(a_perf['Wins']))
                m2.metric("Draws", int(a_perf['Draws']))
                m3.metric("Losses", int(a_perf['Losses']))
                
                m4, m5, m6 = st.columns(3)
                m4.metric("Goals Scored", f"{int(a_perf['Scored'])} ({a_perf['Goals Per Game']:.2f}/g)")
                m5.metric("Goals Conceded", f"{int(a_perf['Conceded'])} ({a_perf['Goals Conceded Per Game']:.2f}/g)")
                m6.metric("Shots/Game", f"{a_perf['Shots Per Game']:.2f} ({a_perf['SoT Per Game']:.2f} SoT)")
                
                st.markdown("**Recent Form (Last 5 matches):**")
                rf1, rf2, rf3 = st.columns(3)
                rf1.metric("Form Points", int(a_profile['Points_Last_5']))
                rf2.metric("Form Wins", int(a_profile['Wins_Last_5']))
                rf3.metric("Form Losses", int(a_profile['Losses_Last_5']))

            st.divider()
            st.write("#### Detailed Feature Comparisons")
            
            compare_options = [
                "⚽ Goals (Scored vs Conceded)",
                "🎯 Shots (Total vs On Target)",
                "📐 Corners",
                "🟨 Discipline (Yellow Cards)",
                "📈 ELO & Rest Days"
            ]
            
            selected_comp = st.segmented_control(
                "Select Feature to Compare:",
                options=compare_options,
                default="⚽ Goals (Scored vs Conceded)",
                key="predictor_feature_compare"
            )
            
            if selected_comp == "⚽ Goals (Scored vs Conceded)":
                goals_compare = pd.DataFrame({
                    "Metric": ["Goals Scored", "Goals Conceded"],
                    home_team: [h_perf['Scored'], h_perf['Conceded']],
                    away_team: [a_perf['Scored'], a_perf['Conceded']]
                }).set_index("Metric")
                st.bar_chart(goals_compare, color=[get_team_color(home_team), get_team_color(away_team)], stack=False)
                
            elif selected_comp == "🎯 Shots (Total vs On Target)":
                shots_compare = pd.DataFrame({
                    "Metric": ["Total Shots", "Shots On Target"],
                    home_team: [h_perf['Total Shots'], h_perf['Shots On Target']],
                    away_team: [a_perf['Total Shots'], a_perf['Shots On Target']]
                }).set_index("Metric")
                st.bar_chart(shots_compare, color=[get_team_color(home_team), get_team_color(away_team)], stack=False)
                
            elif selected_comp == "📐 Corners":
                corners_compare = pd.DataFrame({
                    "Metric": ["Corners Won", "Corners Conceded"],
                    home_team: [h_perf['Corners'], h_perf['Corners_conceded']],
                    away_team: [a_perf['Corners'], a_perf['Corners_conceded']]
                }).set_index("Metric")
                st.bar_chart(corners_compare, color=[get_team_color(home_team), get_team_color(away_team)], stack=False)
                
            elif selected_comp == "🟨 Discipline (Yellow Cards)":
                cards_compare = pd.DataFrame({
                    "Metric": ["Yellow Cards", "Opposing Yellow Cards"],
                    home_team: [h_perf['Yellow Cards'], h_perf['Opposing_Yellow_Cards']],
                    away_team: [a_perf['Yellow Cards'], a_perf['Opposing_Yellow_Cards']]
                }).set_index("Metric")
                st.bar_chart(cards_compare, color=[get_team_color(home_team), get_team_color(away_team)], stack=False)
                
            elif selected_comp == "📈 ELO & Rest Days":
                elo_compare = pd.DataFrame({
                    "Metric": ["ELO Score", "Total Points", "Days Rest"],
                    home_team: [h_profile['ELO Rating'], h_profile['Total_Points'], h_profile['Days_Rest']],
                    away_team: [a_profile['ELO Rating'], a_profile['Total_Points'], a_profile['Days_Rest']]
                }).set_index("Metric")
                st.bar_chart(elo_compare, color=[get_team_color(home_team), get_team_color(away_team)], stack=False)
