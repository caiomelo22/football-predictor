import numpy as np
import pandas as pd


def get_winner(home_score, away_score):
    if home_score is None or away_score is None:
        return None
    elif home_score > away_score:
        return "H"
    elif away_score > home_score:
        return "A"
    else:
        return "D"


def get_games_results(games, scenario):
    loser = "A" if scenario == "H" else "H"
    return (
        len(games.loc[games["winner"] == scenario]),
        len(games.loc[games["winner"] == "D"]),
        len(games.loc[games["winner"] == loser]),
    )


def get_stats_mean(games, n_last_games, scenario):
    games = games.iloc[-n_last_games:, :]

    prefixes = ["team_", "opp_"]
    filtered_cols = [
        col
        for col in games.columns
        if any(col.startswith(prefix) for prefix in prefixes)
        and games[col].dtype in (float, "int32", "int64")
    ]
    team_stats_dict = games[filtered_cols].mean().to_dict()

    return team_stats_dict


def drop_total_games_keys(historical_dict):
    key_to_drop = [
        "home_wins",
        "home_draws",
        "home_losses",
        "away_wins",
        "away_draws",
        "away_losses",
    ]
    historical_dict = {
        key: value for key, value in historical_dict.items() if key not in key_to_drop
    }
    return historical_dict


def get_historical_stats(games, home_games, away_games, suffix=""):
    total_games = len(games)
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
    team, season, game_date, scenario, n_last_games, fixtures_df
):
    response = get_team_previous_games(team, game_date, season, fixtures_df)
    if not response:
        return None

    (
        previous_season_games,
        home_previous_season_games,
        away_previous_season_games,
    ) = response

    total_games = len(previous_season_games.index)
    if (
        total_games < 10
        or (len(home_previous_season_games.index) < 5 and scenario == "H")
        or (len(away_previous_season_games.index) < 5 and scenario == "A")
    ):
        return

    games_dict = get_historical_stats(
        previous_season_games, home_previous_season_games, away_previous_season_games
    )

    previous_last_games = previous_season_games.iloc[-n_last_games:, :]
    home_last_games = previous_last_games.loc[
        previous_last_games["scenario"] == "H"
    ].copy()
    away_last_games = previous_last_games.loc[
        previous_last_games["scenario"] == "A"
    ].copy()

    last_games_dict = get_historical_stats(
        previous_last_games, home_last_games, away_last_games, suffix="last_games"
    )
    last_games_dict = drop_total_games_keys(last_games_dict)

    if scenario == "H":
        prefix = "home"
        previous_season_games_filtered = home_previous_season_games
    else:
        prefix = "away"
        previous_season_games_filtered = away_previous_season_games

    outcome_pct_dict = {
        f"{prefix}_win_pct": games_dict[f"{prefix}_wins"]
        / len(previous_season_games_filtered),
        f"{prefix}_draw_pct": games_dict[f"{prefix}_draws"]
        / len(previous_season_games_filtered),
        f"{prefix}_loss_pct": games_dict[f"{prefix}_losses"]
        / len(previous_season_games_filtered),
    }
    games_dict = drop_total_games_keys(games_dict)

    team_stats_dict = get_stats_mean(previous_season_games, n_last_games, scenario)

    if any(np.isnan(value) for value in team_stats_dict.values()):
        return None

    return_dict = {
        **games_dict,
        **last_games_dict,
        **outcome_pct_dict,
        **team_stats_dict,
    }
    return_dict = {f"{prefix}_{key}": value for key, value in return_dict.items()}

    return return_dict
