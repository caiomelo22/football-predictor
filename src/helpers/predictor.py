import pandas as pd
from thefuzz import fuzz
from helpers import classification as pf
from helpers import stats as bf
from services import BetExplorerService, MySQLService
import os
from joblib import load
from termcolor import colored

from helpers.file import load_from_file

def load_saved_predictor_utils(league):
    predictor_utils = load(f"../dist/betting/{league}.joblib")

    return predictor_utils

def load_model_data(league):
    model_data = load_saved_predictor_utils(league)

    return model_data

def get_season_games(league, season):
    mysql = MySQLService()

    season_games, teams_elo = bf.initialize_matches(league, season)

    teams_query = f"SELECT DISTINCT(home_team) as team FROM matches WHERE season = {season} and league = '{league}'"

    teams = mysql.execute_query(teams_query)

    return season_games, teams, teams_elo

def get_next_games(country, league):
    # Getting odds for next games
    bet_explorer = BetExplorerService(country, league)

    next_games = bet_explorer.get_next_games()

    return next_games

def get_most_compatible_team(team, teams):
    team_compatibility = teams

    team_compatibility["score"] = team_compatibility.apply(
        lambda x: fuzz.ratio(team, x["team"]),
        axis=1,
    )
    team_compatibility = team_compatibility.sort_values(
        by="score", ascending=False
    ).reset_index(drop=True)
    
    return team_compatibility.iloc[0]["team"]

def get_next_games_stats(game, season, min_games_played, min_games_played_at, season_games, teams_elo, teams):
    home_team_compatible = get_most_compatible_team(game["home_team"], teams)
    game["home_team_translated"] = home_team_compatible
    
    away_team_compatible = get_most_compatible_team(game["away_team"], teams)
    game["away_team_translated"] = away_team_compatible
    
    home_stats_dict = bf.get_team_previous_games_stats(
        game["home_team_translated"], season, game["date"], "H", min_games_played, min_games_played_at, season_games
    )
    if not home_stats_dict:
        return None

    away_stats_dict = bf.get_team_previous_games_stats(
        game["away_team_translated"], season, game["date"], "A", min_games_played, min_games_played_at, season_games
    )
    if not away_stats_dict:
        return None

    game_info_keys = [
        "date",
        "season",
        "home_team_translated",
        "away_team_translated",
        "home_odds",
        "away_odds",
        "draw_odds",
        "result",
        "home_score",
        "away_score",
    ]
    game_info_dict = {key: game.get(key) for key in game_info_keys}

    home_elo = teams_elo.get(game["home_team_translated"])
    away_elo = teams_elo.get(game["away_team_translated"])

    return {**home_stats_dict, **away_stats_dict, **game_info_dict, "home_elo": home_elo, "away_elo": away_elo}

def predict_next_games(pipeline, data_df, filtered_cols):
    X = data_df[filtered_cols]

    odds_cols = [
        "date",
        "season",
        "home_team_translated",
        "away_team_translated",
        "home_odds",
        "away_odds",
        "draw_odds",
        "home_elo",
        "away_elo",
    ]
    odds_df = data_df[odds_cols]

    for c in odds_cols:
        if "odds" in c:
            odds_df[c] = pd.to_numeric(odds_df[c], errors="coerce")

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    probs_test_df = pd.DataFrame(
        probabilities,
        index=data_df.index,
        columns=["away_probs", "draw_probs", "home_probs"],
    )
    preds_test_df = pd.DataFrame(predictions, index=data_df.index, columns=["pred"])
    predictions_df = pd.concat([preds_test_df, probs_test_df, odds_df], axis=1)

    predictions_df.dropna(subset=["home_odds"], inplace=True)
    predictions_df = predictions_df[predictions_df["home_odds"] != " "]

    predictions_df = predictions_df.astype(
        {"home_odds": float, "draw_odds": float, "away_odds": float}
    )

    return predictions_df

def get_bets(predictions_df, min_odds, bankroll, strategy, default_value, default_bankroll_pct):
    bets = []
    
    for _, game in predictions_df.iterrows():
        bet_value = 1 # pf.get_bet_value_by_row(game, bankroll, strategy)
        odds, probs = pf.get_bet_odds_probs(game)

        pred_odds = 1/probs
        
        bet_worth_it = pf.classification_bet_worth_it(
            game["pred"],
            odds,
            pred_odds,
            min_odds,
            bet_value
        )

        bet_value = pf.get_bet_unit_value(odds, probs, bankroll, strategy, default_value, default_bankroll_pct)

        pred_str = "Draw"
        if game['pred'] == "H":
            pred_str = game['home_team_translated']
        elif game['pred'] == "A":
            pred_str = game['away_team_translated']

        bet_str = f"{game['home_team_translated']} ({round(game['home_elo'], 2)}) x ({round(game['away_elo'], 2)}) {game['away_team_translated']}: ${bet_value} on {pred_str} @ {odds}"

        if bet_worth_it:
            bets.append(colored(bet_str, "green"))
        elif abs(odds - min_odds) < 0.1:
            bets.append(colored(bet_str, "yellow"))
        else:
            bets.append(colored(bet_str, "red"))

    return bets
    