filtered_cols = [
    'home_points_pct', 
    'home_win_pct', 'home_draw_pct', 'home_loss_pct',
    'home_points_pct_last_games', 'home_win_pct_last_games', 'home_draw_pct_last_games', 'home_loss_pct_last_games',
    'home_home_win_pct', 'home_home_draw_pct', 'home_home_loss_pct',
    # 'home_team_xg', 'home_opp_xg',
    'home_team_score', 'home_opp_score',
    # 'home_home_team_xg', 'home_home_opp_xg',
    'home_home_team_score', 'home_home_opp_score',
    'away_points_pct', 
    'away_win_pct', 'away_draw_pct', 'away_loss_pct',
    'away_points_pct_last_games', 'away_win_pct_last_games', 'away_draw_pct_last_games', 'away_loss_pct_last_games',
    'away_away_win_pct', 'away_away_draw_pct', 'away_away_loss_pct',
    # 'away_team_xg', 'away_opp_xg',
    'away_team_score', 'away_opp_score',
    # 'away_away_opp_xg', 'away_away_team_xg',
    'away_away_opp_score', 'away_away_team_score',
    'home_odds', 'away_odds', 'draw_odds', 
    'home_elo', 'away_elo'
]

selected_stats = {
    "shooting": [
        "Sh",
        "SoT",
        "SoT_pct",
        "G_Sh",
        "G_SoT",
        "Dist",
        "npxG",
        "npxG_Sh",
        "G_xG",
        "np_G_xG",
    ],
    "passing_types": ["Att", "Cmp", "TB", "Sw", "Crs", "CK"],
    "gca": ["SCA"],
    "possession": ["Poss", "Att", "Succ", "Succ_pct"],
    "misc": [
        "CrdY",
        "CrdR",
        "Recov",
    ],
}

strategy = "bankroll_pct"
