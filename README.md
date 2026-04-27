# **Football-Match-Predictions**
#### In this repository, I utilize datasets I got from football-data.co.uk (https://football-data.co.uk/mmz4281/2526/E0.csv) .The datasets contain data of all the Premier League, Bundesliga, LaLiga, Greek Super League matches played until this day.It contains a lot of categories that I explain below.Most of these categories are used in the feature engineering part (src/Preprocessing.py) in order to create more useful categories and after that, they are dropped do avoid data leakage.
The raw data (datasets/Raw_Data/) contain lots of features, most of which are dropped. The features used to clean the data are the ones below.
### Explanation of the dataset categories as per football-data.co.uk:
* Div = League Division
* Date = Match Date 
* Time = Time of match kick off
* HomeTeam = Home Team
* AwayTeam = Away Team
* FTHG = Full Time Home Team Goals
* FTAG  = Full Time Away Team Goals
* FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
* HTHG = Half Time Home Team Goals
* HTAG = Half Time Away Team Goals
* HTR = Half Time Result (H=Home Win, D=Draw, A=Away Win)

Match Statistics (where available)
* Referee = Match Referee
* HS = Home Team Shots
* AS = Away Team Shots
* HST = Home Team Shots on Target
* AST = Away Team Shots on Target
* HHW = Home Team Hit Woodwork
* AHW = Away Team Hit Woodwork
* HC = Home Team Corners
* AC = Away Team Corners
* HF = Home Team Fouls Committed
* AF = Away Team Fouls Committed
* HFKC = Home Team Free Kicks Conceded
* AFKC = Away Team Free Kicks Conceded
* HO = Home Team Offsides
* AO = Away Team Offsides
* HY = Home Team Yellow Cards
* AY = Away Team Yellow Cards
* HR = Home Team Red Cards
* AR = Away Team Red Cards
* HBP = Home Team Bookings Points (10 = yellow, 25 = red)
* ABP = Away Team Bookings Points (10 = yellow, 25 = red)

### All of the above are dropped , and transformed into categories such as:
* Home Team Days Rest
* Away Team Days Rest
* Home Team Average Goals Scored Last 5 Games
* Away Team Average Goals Scored Last 5 Games
* Home Team Average Goals Conceded Last 5 Games
* Away Team Average Goals Conceded Last 5 Games
* Home Team Average Shots Last 5 Games
* Away Team Average Shots Last 5 Games
* Home Team Average Shots Conceded Last 5 Games
* Away Team Average Shots Conceded Last 5 Games
* Home Team Wins Last 5 Games
* Away Team Wins Last 5 Games
* Home Team Losses Last 5 Games
* Away Team Losses Last 5 Games
* Home Team Draws Last 5 Games
* Away Team Draws Last 5 Games
* Home Team ELO Rating
* Away Team ELO Rating
* Home Team Total Points
* Away Team Total Points
* Home Team Points Last 5 Games
* Away Team Points Last 5 Games
* Total Points Difference
* ELO Score Difference
### The features above were used to train the respective models of the 4 Leagues.

## Models
### There are 2 models trained for each league. 
* Random Forest Classifier
* XGBoost Classifier which is used to create the final Calibrated XGBoost model.
#### *For the LaLiga predictions there are some changes regarding the random forest classifier (class_weight='balanced')*

## Streamlit App
### The app.py represents a streamlit application where people can try the models and make predictions. 
1. Choose a League of your preference
2. Choose a Home Team
3. Choose an Away Team
4. Click "Predict" and see the results.
5. Below the predictions there is a bar chart which visualizes the choosed team's season stats by comparing them.
6. The "Switch" button, lets swap the Home Team and the Away Team
