import pandas as pd
import utils.builder_functions as bf
import utils.classification.predictor_functions as pf
import utils.scraper_functions as sf
from datetime import datetime as dt, timedelta
from utils.leagues_info import leagues
import json
import os

league = 'serie-a'
league_info = leagues[league]
n_last_games = 5
bankroll = 1478.25

options_path = f"leagues/{league}/official/columns.json"
with open(options_path, 'r') as json_file:
    options_info = json.load(json_file)

features_kmeans_list, kmeans_scaler_list, pca_features, pca_scaler, pca, pipeline = pf.load_saved_utils(league)

# Getting next games
print('Scrapping fbref...')
data_model, seasons_squad_ids = sf.scrape_fbref(league_info['league_fbref'], league_info['league_id_fbref'])

# Getting advanced stats
print('Scrapping fbref advanced stats...')
games_stats_dict = sf.scrape_advanced_stats(league_info['league_id_fbref'], league_info['season_test'], seasons_squad_ids, options_info['selected_stats'])

# Merging everything
columns = ['date', 'week', 'home_team', 'home_xg', 'home_score', 'away_score', 'away_xg', 'away_team']
complete_games = [sf.complete_stats(game_stats, columns, games_stats_dict) for game_stats in data_model]
season_games = pd.DataFrame(complete_games)

season_games['season'] = league_info['season_test']
exclude_cols = ['date', 'week', 'home_team', 'away_team', 'season']
season_games = season_games.apply(lambda x: pd.to_numeric(x, errors='coerce') if x.name not in exclude_cols else x)
season_games['date'] = pd.to_datetime(season_games['date'])
season_games['winner'] = season_games.apply(lambda x: bf.get_winner(x['home_score'], x['away_score']), axis=1)

# Filter for today and tomorrow's games
today = dt.now()
today_date = today.date()
tomorrow_date = (today + timedelta(days=1)).date()
next_games = season_games[(season_games['date'].dt.date >= today_date) & (season_games['date'].dt.date <= tomorrow_date)].reset_index(drop=True)

# Getting odds for next games
print('Scrapping BetExplorer...')
sf.scrape_betexplorer(next_games, league_info['league_betexplorer'], league_info['country_betexplorer'])

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

path = f"dist"
if not os.path.exists(path):
    os.makedirs(path)
season_games.to_csv(f'{path}/season_games.csv')
data_df.to_csv(f'{path}/data_df.csv')
next_games.to_csv(f'{path}/next_games.csv')

X, _, odds = pf.separate_dataset_info(data_df)

X = pf.apply_kmeans(X, kmeans_scaler_list, features_kmeans_list)
X = pf.apply_pca(X, pca_scaler, pca, pca_features)
predictions = pipeline.predict(X)
probabilities = pipeline.predict_proba(X)

probs_test_df = pd.DataFrame(probabilities, index=data_df.index, columns=['away_probs', 'draw_probs', 'home_probs'])
preds_test_df = pd.DataFrame(predictions, index=data_df.index, columns=['pred'])
test_results_df = pd.concat([preds_test_df, probs_test_df, next_games], axis=1)

test_results_df.dropna(subset=['home_odds'], inplace=True)
test_results_df = test_results_df[test_results_df['home_odds'] != " "]

test_results_df = test_results_df.astype({'home_odds': float, 'draw_odds': float, 'away_odds': float})

today_bets = 0
for _, game in test_results_df.iterrows():
    bet_value = pf.get_bet_value_by_row(game, bankroll, options_info['strategy'])
    odds, probs = pf.get_bet_odds_probs(game)
    bet_worth_it = pf.bet_worth_it(bet_value, odds)
    if not bet_worth_it: continue
    today_bets += 1

    print(f"\n{game['home_team']} ({game['home_odds']})")
    print(f"X ({game['draw_odds']})")
    print(f"{game['away_team']} ({game['away_odds']})")
    print(f"Prediction: {game['pred']} ({odds})")
    print(f"Bet Value: ${round(bet_value, 2)}")

if not today_bets:
    print("\nSorry, there are no bets for today.")