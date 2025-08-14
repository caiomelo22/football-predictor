import numpy as np
import pandas as pd
from tqdm import tqdm
from helpers import elo as eh

from services.mysql.service import MySQLService


def get_1x2_result(row):
    if row["home_score"] > row["away_score"]:
        return "H"
    elif row["away_score"] > row["home_score"]:
        return "A"
    else:
        return "D"
    
def get_ahc_result(row):
    if row["away_score"] + row["away_ahc_odds"] > row["home_score"]:
        return "A"
    elif row["home_score"] + -row["home_ahc_odds"] > row["away_score"]:
        return "H"
    else:
        return "P" # Bet push
    
def get_totals_result(row):
    if row["home_score"] + row["away_score"] > row["totals_line"]:
        return "O" # Bet over
    elif row["home_score"] + row["away_score"] < row["totals_line"]:
        return "U" # Bet under
    else:
        return "P"  # Bet push


def initialize_matches(league, start_season):
    mysql_service = MySQLService()

    where_clause = f"league = '{league}'"
    order_by_clause = "date ASC"
    data = mysql_service.get_data("matches_v2", where_clause=where_clause, order_by_clause=order_by_clause)

    data["date"] = pd.to_datetime(data["date"])
    data = data.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)

    data["result"] = data.apply(get_1x2_result, axis=1)
    data["ahc_result"] = data.apply(get_ahc_result, axis=1)
    data["totals_result"] = data.apply(get_totals_result, axis=1)

    # Initialize ELO ratings for teams
    data["home_elo"] = 1500
    data["away_elo"] = 1500

    teams_elo = initialize_elo_dict(data)

    print("Generating teams ELOs...")

    for index in tqdm(range(len(data))):
        row = data.iloc[index]

        data.at[index, "home_elo"] = teams_elo[row["home_team"]]
        data.at[index, "away_elo"] = teams_elo[row["away_team"]]

        teams_elo = eh.update_match_results(
            teams_elo,
            row["home_team"],
            row["away_team"],
            row["home_score"],
            row["away_score"],
        )

    print("Successfully generated teams ELOs.")

    return data[data["season"] >= start_season], teams_elo


def initialize_elo_dict(matches):
    teams = set(
        matches["home_team"].unique().tolist() + matches["away_team"].unique().tolist()
    )
    teams_elo = {team: 1500 for team in teams}
    return teams_elo


def get_games_results(games, scenario):
    loser = "A" if scenario == "H" else "H"
    return (
        len(games.loc[games["result"] == scenario]),
        len(games.loc[games["result"] == "D"]),
        len(games.loc[games["result"] == loser]),
    )


def get_stats_mean(games, n_last_games, prefix=""):
    games = games.iloc[-n_last_games:, :]

    prefixes = ["team_", "opp_"]
    filtered_cols = [
        col
        for col in games.columns
        if any(col.startswith(prefix) for prefix in prefixes)
        and games[col].dtype in (float, "int32", "int64")
        and "_odds" not in col
        and "_elo" not in col
    ]
    team_stats_dict: dict = games[filtered_cols].mean().to_dict()

    if prefix:
        team_stats_dict = {f"{prefix}_{key}": value for key, value in team_stats_dict.items()}

    return team_stats_dict


def drop_total_games_keys(historical_dict, suffix=""):
    if suffix:
        suffix = f"_{suffix}"

    key_to_drop = [
        f"home_wins{suffix}",
        f"home_draws{suffix}",
        f"home_losses{suffix}",
        f"away_wins{suffix}",
        f"away_draws{suffix}",
        f"away_losses{suffix}",
    ]

    historical_dict = {
        key: value for key, value in historical_dict.items() if key not in key_to_drop
    }

    return historical_dict


def get_historical_stats(home_games, away_games, suffix=""):
    total_games = len(home_games) + len(away_games)
    home_wins, home_draws, home_losses = get_games_results(home_games, "H")
    away_wins, away_draws, away_losses = get_games_results(away_games, "A")

    total_wins = home_wins + away_wins
    total_draws = home_draws + away_draws
    total_losses = home_losses + away_losses

    win_pct = total_wins / total_games
    draw_pct = total_draws / total_games
    loss_pct = total_losses / total_games

    points_achieved = total_wins * 3 + total_draws
    points_pct = points_achieved / (total_games * 3)

    historical_keys = [
        "points_pct",
        "win_pct",
        "draw_pct",
        "loss_pct",
        "home_wins",
        "home_draws",
        "home_losses",
        "away_wins",
        "away_draws",
        "away_losses",
    ]
    historical_vals = [
        points_pct,
        win_pct,
        draw_pct,
        loss_pct,
        home_wins,
        home_draws,
        home_losses,
        away_wins,
        away_draws,
        away_losses,
    ]

    historical_dict = dict(zip(historical_keys, historical_vals))
    if suffix:
        historical_dict = {
            f"{key}_{suffix}": value for key, value in historical_dict.items()
        }

    return historical_dict


def get_team_previous_games(team, game_date, season, fixtures_df):
    home_previous_games = fixtures_df.loc[
        (fixtures_df["home_team"] == team) & (fixtures_df["date"] < game_date)
    ].copy()
    away_previous_games = fixtures_df.loc[
        (fixtures_df["away_team"] == team) & (fixtures_df["date"] < game_date)
    ].copy()

    if len(home_previous_games.index) == 0 or len(away_previous_games.index) == 0:
        return None

    home_games_cols_renamed = {
        col: col.replace("home_", "team_").replace("away_", "opp_")
        for col in home_previous_games.columns
    }
    away_games_cols_renamed = {
        col: col.replace("away_", "team_").replace("home_", "opp_")
        for col in away_previous_games.columns
    }

    home_previous_games.rename(columns=home_games_cols_renamed, inplace=True)
    home_previous_games["scenario"] = "H"

    away_previous_games.rename(columns=away_games_cols_renamed, inplace=True)
    away_previous_games["scenario"] = "A"

    previous_games = pd.concat(
        [home_previous_games, away_previous_games], axis=0, ignore_index=True
    )
    previous_games.sort_values("date", inplace=True)

    previous_season_games = previous_games.loc[
        previous_games["season"] == season
    ].copy()
    home_previous_season_games = home_previous_games.loc[
        home_previous_games["season"] == season
    ].copy()
    away_previous_season_games = away_previous_games.loc[
        away_previous_games["season"] == season
    ].copy()

    return previous_season_games, home_previous_season_games, away_previous_season_games


def get_team_previous_games_stats(
    team, season, game_date, scenario, n_last_games, n_last_games_at, fixtures_df
):
    response = get_team_previous_games(team, game_date, season, fixtures_df)
    if not response:
        return None

    (
        previous_season_games,
        home_previous_season_games,
        away_previous_season_games,
    ) = response

    total_games = len(previous_season_games)
    if (
        total_games < n_last_games
        or (len(home_previous_season_games) < n_last_games_at and scenario == "H")
        or (len(away_previous_season_games) < n_last_games_at and scenario == "A")
    ):
        return

    whole_season_dict = get_historical_stats(
        home_previous_season_games, away_previous_season_games
    )

    home_last_games = home_previous_season_games.loc[
        home_previous_season_games["scenario"] == "H"
    ].iloc[-n_last_games_at:, :]
    away_last_games = away_previous_season_games.loc[
        away_previous_season_games["scenario"] == "A"
    ].iloc[-n_last_games_at:, :]

    last_games_dict = get_historical_stats(
        home_last_games, away_last_games, suffix="last_games"
    )
    last_games_dict = drop_total_games_keys(last_games_dict, suffix="last_games")

    if scenario == "H":
        prefix = "home"
        previous_season_games_filtered = home_previous_season_games
    else:
        prefix = "away"
        previous_season_games_filtered = away_previous_season_games

    outcome_pct_dict = {
        f"{prefix}_win_pct": whole_season_dict[f"{prefix}_wins"]
        / len(previous_season_games_filtered),
        f"{prefix}_draw_pct": whole_season_dict[f"{prefix}_draws"]
        / len(previous_season_games_filtered),
        f"{prefix}_loss_pct": whole_season_dict[f"{prefix}_losses"]
        / len(previous_season_games_filtered),
    }
    whole_season_dict = drop_total_games_keys(whole_season_dict)

    team_stats_dict_filtered = get_stats_mean(previous_season_games_filtered, n_last_games_at, prefix=prefix)

    team_stats_dict = get_stats_mean(previous_season_games, n_last_games_at)

    return_dict = {
        **whole_season_dict,
        **last_games_dict,
        **outcome_pct_dict,
        **team_stats_dict,
        **team_stats_dict_filtered
    }
    return_dict = {f"{prefix}_{key}": value for key, value in return_dict.items()}

    return return_dict
