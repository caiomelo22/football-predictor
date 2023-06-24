import pandas as pd
import utils.builder_functions as bf
import utils.predictor_functions as pf
import utils.scraper_functions as sf
from datetime import datetime as dt, timedelta

league_fbref = 'Major-League-Soccer'
league_id_fbref = 22
league_betexplorer = 'mls'
utils_league = 'major-league-soccer'
country_betexplorer = 'usa'
seasons = '2018-2024'
season_test = 2023
n_last_games = 5

features_kmeans_list, kmeans_scaler_list, pca_features, pca_scaler, pca, pipeline = pf.load_saved_utils(utils_league)

# Getting next games
print('Scrapping fbref...')
data_model, seasons_squad_ids = sf.scrape_fbref(league_fbref, league_id_fbref)

# Getting advanced stats
print('Scrapping fbref advanced stats...')
games_stats_dict = sf.scrape_advanced_stats(league_id_fbref, season_test, seasons_squad_ids)

# Merging everything
columns = ['date', 'week', 'home_team', 'home_xg', 'home_score', 'away_score', 'away_xg', 'away_team']
complete_games = [sf.complete_stats(game_stats, columns, games_stats_dict) for game_stats in data_model]
season_games = pd.DataFrame(complete_games)
season_games['season'] = season_test
season_games['winner'] = season_games.apply(lambda x: bf.get_winner(x['home_score'], x['away_score']), axis=1)

exclude_cols = ['date', 'week', 'home_team', 'away_team', 'season']
season_games = season_games.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name not in exclude_cols else x)
season_games['date'] = pd.to_datetime(season_games['date'])

# Filter for today and tomorrow's games
today = dt.now()
today_date = today.date()
tomorrow_date = (today + timedelta(days=1)).date()
next_games = season_games[(season_games['date'].dt.date >= today_date) & (season_games['date'].dt.date <= tomorrow_date)].reset_index(drop=True)

# Getting odds for next games
print('Scrapping BetExplorer...')
sf.scrape_betexplorer(next_games, league_betexplorer, country_betexplorer)

data_model = []
for _, game in next_games.iterrows():
    home_stats_dict = bf.get_team_previous_games_stats(game['home_team'], game['season'], game['date'], 'H', n_last_games, season_games)
    if not home_stats_dict:
        continue
        
    away_stats_dict = bf.get_team_previous_games_stats(game['away_team'], game['season'], game['date'], 'A', n_last_games, season_games)
    if not away_stats_dict:
        continue
        
    game_info_keys = ['date', 'season', 'home_team', 'away_team', 'home_odds', 'away_odds', 'draw_odds', 'winner', 'home_score', 'away_score']
    game_info_dict = {key: game.get(key) for key in game_info_keys}
        
    data_model.append({**home_stats_dict, **away_stats_dict, **game_info_dict})

data_df = pd.DataFrame(data_model)
data_df.to_csv('testing.csv')
next_games.to_csv('next_testing.csv')

X, _, odds = pf.separate_dataset_info(data_df)

X = pf.apply_kmeans(X, kmeans_scaler_list, features_kmeans_list)
X = pf.apply_pca(X, pca_scaler, pca, pca_features)
predictions = pipeline.predict(X)

next_games['prediction'] = predictions

for _, game in next_games.iterrows():
    print(f"\n{game['home_team']} ({game['home_odds']})")
    print(f"X ({game['draw_odds']})")
    print(f"{game['away_team']} ({game['away_odds']})")
    print(f"Prediction: {game['prediction']}")