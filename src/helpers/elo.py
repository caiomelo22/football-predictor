import math


def expected_result(elo_diff):
    return 1 / (10 ** (-elo_diff / 400) + 1)


def update_elo(elo_winner, elo_loser, result, k):
    expected_winner = expected_result(elo_winner - elo_loser)
    delta = (result - expected_winner) * k
    return delta


def goal_difference_adjustment(delta, margin):
    return delta * math.sqrt(margin) if margin > 0 else delta


def update_tilt(old_tilt, game_total_goals, opposition_tilt, exp_game_total_goals):
    return 0.98 * old_tilt + 0.02 * (
        game_total_goals / opposition_tilt / exp_game_total_goals
    )


# Setting 20 as default value for k
def update_match_results(
    team_elo_ratings, home_team, away_team, home_goals, away_goals, k=20
):
    margin = abs(home_goals - away_goals)
    result = 1 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0

    home_elo = team_elo_ratings[home_team]
    away_elo = team_elo_ratings[away_team]

    delta_home = update_elo(home_elo, away_elo, result, k)
    delta_home_adjusted = goal_difference_adjustment(delta_home, margin)

    team_elo_ratings[home_team] += round(delta_home_adjusted, 2)
    team_elo_ratings[away_team] -= round(delta_home_adjusted, 2)

    return team_elo_ratings
