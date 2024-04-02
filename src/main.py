import pandas as pd
from helpers import classification as pf
from datetime import datetime as dt, timedelta
import json
import os

from helpers.file import load_from_file

league = "major-league-soccer"
league_fbref = "Serie-A"
league_id_fbref = 24
league_betexplorer = "serie-a"
country_betexplorer = "brazil"
seasons = "2019-2024"
season_test = 2023
n_last_games = 5
bankroll = 760.85

path = f"leagues/{league}/official"
options_info = load_from_file(path, 'columns')

pipeline = pf.load_saved_utils(league)

# Getting odds for next games
print("Scrapping BetExplorer...")
sf.scrape_betexplorer(
    next_games, league_info["league_betexplorer"], league_info["country_betexplorer"]
)

data_model = []
for _, game in next_games.iterrows():
    home_stats_dict = bf.get_team_previous_games_stats(
        game["home_team"], game["season"], game["date"], "H", n_last_games, season_games
    )
    if not home_stats_dict:
        continue

    away_stats_dict = bf.get_team_previous_games_stats(
        game["away_team"], game["season"], game["date"], "A", n_last_games, season_games
    )
    if not away_stats_dict:
        continue

    game_info_keys = [
        "date",
        "season",
        "home_team",
        "away_team",
        "home_odds",
        "away_odds",
        "draw_odds",
        "winner",
        "home_score",
        "away_score",
    ]
    game_info_dict = {key: game.get(key) for key in game_info_keys}

    data_model.append({**home_stats_dict, **away_stats_dict, **game_info_dict})

data_df = pd.DataFrame(data_model)

path = f"dist"
if not os.path.exists(path):
    os.makedirs(path)
season_games.to_csv(f"{path}/season_games.csv")
data_df.to_csv(f"{path}/data_df.csv")
next_games.to_csv(f"{path}/next_games.csv")

X, _, odds = pf.separate_dataset_info(data_df)

predictions = pipeline.predict(X)
probabilities = pipeline.predict_proba(X)

probs_test_df = pd.DataFrame(
    probabilities,
    index=data_df.index,
    columns=["away_probs", "draw_probs", "home_probs"],
)
preds_test_df = pd.DataFrame(predictions, index=data_df.index, columns=["pred"])
test_results_df = pd.concat([preds_test_df, probs_test_df, next_games], axis=1)

test_results_df.dropna(subset=["home_odds"], inplace=True)
test_results_df = test_results_df[test_results_df["home_odds"] != " "]

test_results_df = test_results_df.astype(
    {"home_odds": float, "draw_odds": float, "away_odds": float}
)

today_bets = 0
for _, game in test_results_df.iterrows():
    bet_value = pf.get_bet_value_by_row(game, bankroll, options_info["strategy"])
    odds, probs = pf.get_bet_odds_probs(game)
    bet_worth_it = pf.bet_worth_it(bet_value, odds)
    if bet_value < 0:
        continue
    today_bets += 1

    print(f"\n{game['home_team']} ({game['home_odds']})")
    print(f"X ({game['draw_odds']})")
    print(f"{game['away_team']} ({game['away_odds']})")
    print(f"Prediction: {game['pred']} ({odds})")
    print(f"Bet Value: ${round(bet_value, 2)}")
    print(f"{'GOOD' if bet_worth_it else 'BAD'} BET")

if not today_bets:
    print("\nSorry, there are no bets for today.")
