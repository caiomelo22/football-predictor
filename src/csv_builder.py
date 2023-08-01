import os
import pandas as pd
from dotenv import load_dotenv
from utils import builder_functions as bf
from services import MySQLService
import warnings
warnings.filterwarnings('ignore')

# Load environment variables from the .env file
load_dotenv()

league = 'serie-a'
start_season = 2019
end_season = 2024
n_last_games = 5

seasons = []
mysql_service = MySQLService()

for season in range(start_season, end_season):
    where_clause=f"league = '{league}' AND season = {season}"
    data = mysql_service.get_data("matches", where_clause=where_clause)
    seasons.append(data)
    seasons[-1]['season'] = season

fixtures_df = pd.concat(seasons, axis=0).reset_index(drop=True)
fixtures_df.dropna(subset=['home_odds', 'away_odds', 'draw_odds'], inplace=True)
fixtures_df['home_odds'] = fixtures_df['home_odds'].astype(float)
fixtures_df['away_odds'] = fixtures_df['away_odds'].astype(float)
fixtures_df['draw_odds'] = fixtures_df['draw_odds'].astype(float)
fixtures_df['winner'] = fixtures_df.apply(lambda x: bf.get_winner(x['home_score'], x['away_score']), axis=1)

data_model = []

for index, game in fixtures_df.iterrows():
    
    print("{}/{}".format(index, len(fixtures_df)))
    
    if pd.isnull(game['home_odds']):
        continue
    
    home_stats_dict = bf.get_team_previous_games_stats(game['home_team'], game['season'], game['date'], 'H', n_last_games, fixtures_df)
    if not home_stats_dict:
        continue
        
    away_stats_dict = bf.get_team_previous_games_stats(game['away_team'], game['season'], game['date'], 'A', n_last_games, fixtures_df)
    if not away_stats_dict:
        continue
        
    game_info_keys = ['date', 'season', 'home_team', 'away_team', 'home_odds', 'away_odds', 'draw_odds', 'winner', 'home_score', 'away_score']
    game_info_dict = {key: game[key] for key in game_info_keys}
        
    data_model.append({**home_stats_dict, **away_stats_dict, **game_info_dict})

data_df = pd.DataFrame(data_model)

def parse_df_to_csv(dataframe, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    dataframe.to_csv("{}/{}".format(path, filename))

parse_df_to_csv(data_df, f'leagues/{league}/formatted_data', '{}-{}.csv'.format(start_season, end_season))