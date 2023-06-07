import pandas as pd
import utils.builder_functions as bf
import utils.predictor_functions as pf
import utils.scraper_functions as sf
from datetime import datetime as dt, timedelta

league_fbref = 'Major-League-Soccer'
league_id_fbref = 22
league_betexplorer = 'mls'
country_betexplorer = 'usa'
seasons = '2018-2024'
season_test = 2023
n_last_games = 5

features_kmeans_list, kmeans_scaler_list, pca_features, pca_scaler, pca, pipeline = pf.load_saved_utils(league_betexplorer)

# Getting next games
print('Scrapping fbref...')
season_games = sf.scrape_fbref(league_fbref, league_id_fbref)
season_games['season'] = season_test
season_games['winner'] = season_games.apply(lambda x: bf.get_winner(x['home_score'], x['away_score']), axis=1)

# Getting odds for next games
print('Scrapping BetExplorer...')
sf.scrape_betexplorer(season_games, league_betexplorer, country_betexplorer)

numerical_cols = ['home_xg','home_score','away_score','away_xg']
season_games[numerical_cols] = season_games[numerical_cols].apply(pd.to_numeric)

# Filter for today and tomorrow's games
today = dt.now()
today_date = today.date()
tomorrow_date = (today + timedelta(days=1)).date()
next_games = season_games[(season_games['date'].dt.date >= today_date) & (season_games['date'].dt.date <= tomorrow_date)].reset_index(drop=True)

data_model = []
for _, game in next_games.iterrows():
    home_stats = bf.get_team_previous_games_stats(game['home_team'], game['season'], game['date'], 'H', n_last_games, season_games)
    if not home_stats:
        continue
        
    away_stats = bf.get_team_previous_games_stats(game['away_team'], game['season'], game['date'], 'A', n_last_games, season_games)
    if not away_stats:
        continue
        
    data_model.append([game['date'], game['season'], game['home_team'], game['away_team'], game['home_odds'], game['away_odds'], game['draw_odds']] + home_stats + away_stats + [None, None, None])

data_df = bf.build_formatted_csv(data_model)

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