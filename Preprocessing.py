import pandas as pd
import numpy as np
def __init__():
   return
def preprocess(dataset,league):
    try:
        dataset=dataset.drop(columns=['Div','HR','AR','Time','Referee'])
    except:
       dataset=dataset.drop(columns=['Div','HR','AR','Time'])

    
    dataset['Date'] = pd.to_datetime(dataset['Date'], format='%d/%m/%Y')
    dataset['Month']=dataset['Date'].dt.month
    dataset['Day']=dataset['Date'].dt.dayofweek
    
    

    home_matches=dataset[['Date','HomeTeam']].rename(columns={'HomeTeam':'Team'})
    away_matches=dataset[['Date','AwayTeam']].rename(columns={'AwayTeam':'Team'})

    matches=pd.concat((home_matches,away_matches)).sort_values(['Team','Date'])

    matches['Days_Rest']=matches.groupby('Team')['Date'].diff().dt.days
    matches['Days_Rest']=matches['Days_Rest'].fillna(15)

    dataset=dataset.merge(matches, left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left')
    dataset=dataset.rename(columns={'Days_Rest': 'Home_Days_Rest'}).drop(columns=['Team'])

    dataset=dataset.merge(matches, left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    dataset=dataset.rename(columns={'Days_Rest': 'Away_Days_Rest'}).drop(columns=['Team'])

    dataset=dataset.sort_values('Date').reset_index(drop=True)


    home_stats = dataset[['Date', 'HomeTeam', 'FTHG', 'FTAG','HS','AS','HST',
                      'AST','HC','AC','HF','AF','HY','AY','Home_Days_Rest']].copy()
    home_stats = home_stats.rename(columns={
    'HomeTeam':'Team',
    'FTHG':'Scored',
    'FTAG':'Conceded',
    'HS':'Shots_made',
    'AS':'Shots_Conceded',
    'HST':'Shots_ontarget_made',
    'AST':'Shots_ontarget_conceded',
    'HC':'Corners_made',
    'AC':'Corners_conceded',
    'HF':'Fouls_commited',
    'AF':'Fouls_suffered',
    'HY':'Yellow_Cards',
    'AY':'Opposing_Yellow_Cards',
    'Home_Days_Rest':'Days_Rest'
    })
    home_stats['Wins']= (dataset['FTR']=='H').astype(int)
    home_stats['Draws']=(dataset['FTR']=='D').astype(int)
    home_stats['Losses']=(dataset['FTR']=='A').astype(int)

    away_stats = dataset[['Date','AwayTeam','FTAG','FTHG','HS','AS','HST',
                      'AST','HC','AC','HF','AF','HY','AY','Away_Days_Rest']].copy()
    away_stats = away_stats.rename(columns={
    'AwayTeam':'Team',
    'FTAG':'Scored',
    'FTHG':'Conceded',
    'HS':'Shots_Conceded',#This does not refer to the goals conceded , it contains the amount of all shots against the team
    'AS':'Shots_made',
    'HST':'Shots_ontarget_conceded',
    'AST':'Shots_ontarget_made',
    'HC':'Corners_conceded',#This does not refer to the goals conceded from corners , but the amount of corners the opposing team got
    'AC':'Corners_made',
    'HF':'Fouls_suffered',
    'AF':'Fouls_commited',
    'HY':'Opposing_Yellow_Cards',
    'AY':'Yellow_Cards',
    'Away_Days_Rest':'Days_Rest'
    })

    away_stats['Wins']=(dataset['FTR']=='A').astype(int)
    away_stats['Draws']=(dataset['FTR']=='D').astype(int)
    away_stats['Losses']=(dataset['FTR']=='H').astype(int)

    all_stats=pd.concat([home_stats,away_stats])
    team_performance=all_stats.drop(columns=['Date']).groupby('Team').sum().reset_index()
    total_games=team_performance['Wins']+team_performance['Draws']+team_performance['Losses']


    team_performance['Goal_Difference']=team_performance['Scored']-team_performance['Conceded']
    team_performance['Shots_Per_Game']=team_performance['Shots_made']/total_games
    team_performance['Corners_Per_Game']=team_performance['Corners_made']/total_games
    team_performance['SoT_Per_Game']=team_performance['Shots_ontarget_made']/total_games
    team_performance['Goals_Per_Game']=team_performance['Scored']/total_games
    team_performance['Goals_Conceded_Per_Game']=team_performance['Conceded']/total_games
    team_performance['Fouls_commited_Per_Game']=team_performance['Fouls_commited']/total_games
    team_performance['Shots_Conceded_Per_Game']=team_performance['Shots_Conceded']/total_games
    team_performance['Yellow_Cards_Per_Game']=team_performance['Yellow_Cards']/total_games

    team_stats = pd.concat([home_stats, away_stats]).sort_values(['Team', 'Date']).reset_index(drop=True)


    team_stats['Avg_Scored_Last_5'] = team_stats.groupby('Team')['Scored'].\
    transform(lambda x: x.shift(1).rolling(window=5).mean())
    team_stats['Avg_Conceded_Last_5'] = team_stats.groupby('Team')['Conceded'].\
    transform(lambda x: x.shift(1).rolling(window=5).mean())

    team_stats['Wins_Last_5']=team_stats.groupby('Team')['Wins'].transform\
    (lambda x: x.shift(1).rolling(window=5).sum())
    team_stats['Losses_Last_5']=team_stats.groupby('Team')['Losses'].transform\
     (lambda x:x.shift(1).rolling(window=5).sum())
    team_stats['Draws_Last_5']=team_stats.groupby('Team')['Draws'].transform\
    (lambda x: x.shift(1).rolling(window=5).sum())

    team_stats['Avg_Shots_Last_5']=team_stats.groupby('Team')['Shots_made'].transform\
    (lambda x: x.shift(1).rolling(window=5).mean())
    team_stats['Avg_Shots_Conceded_Last_5']=team_stats.groupby('Team')['Shots_Conceded'].transform\
    (lambda x: x.shift(1).rolling(window=5).mean())

    team_stats['Avg_shots_ontarget_Last_5']=team_stats.groupby('Team')['Shots_ontarget_made'].\
    transform(lambda x: x.shift(1).rolling(window=5).mean())
    team_stats['Avg_shots_ontarget_Conceded_Last_5']=team_stats.groupby('Team')['Shots_ontarget_conceded'].\
    transform(lambda x: x.shift(1).rolling(window=5).mean())

    team_stats['Avg_corners_Last_5']=team_stats.groupby('Team')['Corners_made'].transform\
    (lambda x: x.shift(1).rolling(window=5).mean())
    team_stats['Avg_corners_conceded_Last_5']=team_stats.groupby('Team')['Corners_conceded'].transform\
    (lambda x: x.shift(1).rolling(window=5).mean())


    team_stats=team_stats.fillna(0)

    dataset=dataset.merge(
        team_stats[['Date', 'Team', 'Avg_Scored_Last_5', 'Avg_Conceded_Last_5','Avg_Shots_Last_5',
                'Avg_Shots_Conceded_Last_5',
                'Wins_Last_5',
                'Losses_Last_5']],
        left_on=['Date', 'HomeTeam'],
        right_on=['Date', 'Team'],
        how='left'
    ).rename(columns={
    'Avg_Scored_Last_5': 'Home_Avg_Scored_5',
    'Avg_Conceded_Last_5': 'Home_Avg_Conceded_5',
    'Avg_Shots_Last_5':'Home_Avg_Shots_Last_5',
    'Avg_Shots_Conceded_Last_5':'Home_Avg_Shots_Conceded_Last_5',
    'Wins_Last_5':'Home_Wins_Last_5',
    'Losses_Last_5':'Home_Losses_Last_5'
    }).drop(columns=['Team'])

    dataset=dataset.merge(
    team_stats[['Date', 'Team', 'Avg_Scored_Last_5', 'Avg_Conceded_Last_5','Avg_Shots_Last_5',
                'Avg_Shots_Conceded_Last_5',
                'Wins_Last_5',
                'Losses_Last_5']],
    left_on=['Date', 'AwayTeam'],
    right_on=['Date', 'Team'],
    how='left'
    ).rename(columns={
    'Avg_Scored_Last_5': 'Away_Avg_Scored_5',
    'Avg_Conceded_Last_5': 'Away_Avg_Conceded_5',
    'Avg_Shots_Last_5':'Away_Avg_Shots_Last_5',
    'Avg_Shots_Conceded_Last_5':'Away_Avg_Shots_Conceded_Last_5',
    'Wins_Last_5':'Away_Wins_Last_5',
    'Losses_Last_5':'Away_Losses_Last_5'
    }).drop(columns=['Team'])

    dataset,teams_elo=calculate_team_elo(dataset,league)
    dataset['ELO_Diff']=dataset['Home_ELO_Score']-dataset['Away_ELO_Score']
    dataset=dataset.copy()
    return dataset,team_stats,teams_elo

def expected_probability(elo_a,elo_b):
  return 1/(1+10**((elo_b-elo_a)/400))
     

def elo_rating(elo_a,expected_probability,result,K=20):
  return elo_a+K*(result-expected_probability)
     

def k_Calculation(team_elo,matches_played):
  if team_elo>1800:
    return 10
  elif matches_played>20:
    return 15
  else:
    return 20

def calculate_team_elo(dataset,league="Premier League"):
    if league.lower()=='premier league':
        teams_elo={
        'Man City':1941,
        'Liverpool':1926,
        'Arsenal':2061,
        'Aston Villa':1885,
        'Brighton':1844,
        'Sunderland':1500,
        'Tottenham':1760,
        'Wolves':1640,
        'Nott\'m Forest':1575,
        'Man United':1881,
        'Leeds':1500,
        'Everton':1600,
        'Bournemouth':1750,
        'Newcastle':1800,
        'Chelsea':1869,
        'West Ham':1690,
        'Fulham':1750,
        'Crystal Palace':1800,
        'Burnley':1500
        }
    elif league.lower()=='laliga':
       teams_elo={
        'Real Madrid':1940,
        'Barcelona':1890,
        'Ath Madrid':1850,
        'Villarreal':1720,
        'Betis':1730,
        'Celta':1630,
        'Sociedad':1620,
        'Getafe':1660,
        'Ath Bilbao':1690,
        'Osasuna':1650,
        'Espanyol':1670,
        'Valencia':1701,
        'Girona':1650,
        'Vallecano':1630,
        'Alaves':1560,
        'Sevilla':1690,
        'Elche':1500,
        'Mallorca':1660,
        'Levante':1500,
        'Oviedo':1500
        }
    elif league.lower()=='greek super league':
       teams_elo={
          'Aris':1525,
          'Volos NFC':1375,
          'Olympiakos':1630,
          'AEK':1620,
          'PAOK':1620,
          'Asteras Tripolis':1300,
          'Larisa':1310,
          'Panserraikos':1290,
          'Levadeiakos':1320,
          'Kifisia':1280,
          'Atromitos':1299,
          'Panetolikos':1278,
          'Panathinaikos':1550,
          'OFI Crete':1403
       }
    elif league.lower()=='bundesliga':
       teams_elo={
          'Bayern Munich':1930,
          'RB Leipzig':1720,
          'Ein Frankfurt':1640,
          'Werder Bremen':1597,
          'Freiburg':1644,
          'Wolfsburg':1585,
          'Leverkusen':1820,
          'Hoffenheim':1672,
          'Union Berlin':1576,
          'Stuttgard':1721,
          'St Pauli':1533,
          'Dortmund':1845,
          'Mainz':1631,
          'FC Koln':1529,
          'M\'gladbach':1590,
          'Hamburg':1550,
          'Heidenheim':1530,
          'Augsburg':1564
       }

    default_elo=1500
    home_elo=[]
    away_elo=[]
    matches_played={}

    for index,row in dataset.iterrows():
        home_team=row['HomeTeam']
        away_team=row['AwayTeam']
        result=row['FTR']

        if home_team not in matches_played:
            matches_played[home_team]=0
        if away_team not in matches_played:
            matches_played[away_team]=0

        if home_team not in teams_elo:
            teams_elo[home_team]=default_elo
            matches_played[home_team]=0
        if away_team not in teams_elo:
            teams_elo[away_team]=default_elo
            matches_played[away_team]=0

        current_home_elo=teams_elo[home_team]
        current_away_elo=teams_elo[away_team]
    

        expected_home_elo=expected_probability(current_home_elo,current_away_elo)
        expected_away_elo=expected_probability(current_away_elo,current_home_elo)

        if result=='H':
            home_result,away_result=1,0
        elif result=='D':
            home_result,away_result=0.5,0.5
        else:
            home_result,away_result=0,1

        home_k=k_Calculation(current_home_elo,matches_played[home_team])
        away_k=k_Calculation(current_away_elo,matches_played[away_team])

        teams_elo[home_team]=elo_rating(current_home_elo,expected_home_elo,home_result,home_k)
        teams_elo[away_team]=elo_rating(current_away_elo,expected_away_elo,away_result,away_k)

        matches_played[home_team]+=1
        matches_played[away_team]+=1
        home_elo.append(current_home_elo)
        away_elo.append(current_away_elo)

    dataset['Home_ELO_Score']=home_elo
    dataset['Away_ELO_Score']=away_elo
    return dataset,teams_elo