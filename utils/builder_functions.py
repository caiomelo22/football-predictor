import numpy as np
import pandas as pd

def get_winner(home_score, away_score):
    if home_score > away_score:
        return 'H'
    elif away_score > home_score:
        return 'A'
    else:
        return 'D'
    
def get_games_results(games, scenario):
    loser = 'A' if scenario == 'H' else 'H'
    return len(games.loc[games['winner'] == scenario].index), len(games.loc[games['winner'] == 'D'].index), len(games.loc[games['winner'] == loser].index)

def get_stats_mean(games, n_last_games, scenario):
    games = games.iloc[-n_last_games:,:]
    
    team_stats = [games['team_score'].mean(), games['opp_score'].mean(), games['team_xg'].mean(), games['opp_xg'].mean()]
    
    return team_stats

def get_historical_stats(games, home_games, away_games):
    total_games = len(games.index)
    home_wins, home_draws, home_losses = get_games_results(home_games, 'H')
    away_wins, away_draws, away_losses = get_games_results(away_games, 'A')
    
    total_wins = home_wins + away_wins
    total_draws = home_draws + away_draws
    total_losses = home_losses + away_losses
    
    win_pct = total_wins * 100 / total_games
    draw_pct = total_draws * 100 / total_games
    loss_pct = total_losses * 100 / total_games
    
    points_achieved = total_wins * 3 + total_draws
    points_pct = (points_achieved * 100) / (total_games * 3)
    
    return points_pct, win_pct, draw_pct, loss_pct, home_wins, home_draws, home_losses, away_wins, away_draws, away_losses

def get_team_previous_games(team, game_date, season, fixtures_df):
    home_previous_games = fixtures_df.loc[(fixtures_df['home_team'] == team) & (fixtures_df['date'] < game_date)]
    away_previous_games = fixtures_df.loc[(fixtures_df['away_team'] == team) & (fixtures_df['date'] < game_date)]
    
    if len(home_previous_games.index) == 0 or len(away_previous_games.index) == 0:
        return None
    
    home_previous_games.rename(columns = {
        'home_team': 'team', 'home_score': 'team_score', 'home_xg': 'team_xg',
                                          
        'away_team': 'opp', 'away_score': 'opp_score', 'away_xg': 'opp_xg',
                                          
        'home_odds': 'team_odds', 'away_odds': 'opp_odds'}, inplace=True)
    home_previous_games['scenario'] = 'H'
    
    away_previous_games.rename(columns = {
        'away_team': 'team', 'away_score': 'team_score', 'away_xg': 'team_xg',
                                          
        'home_team': 'opp', 'home_score': 'opp_score', 'home_xg': 'opp_xg',
                                          
        'away_odds': 'team_odds', 'home_odds': 'opp_odds'}, inplace=True)
    away_previous_games['scenario'] = 'A'
    
    previous_games = pd.concat([home_previous_games, away_previous_games], axis=0, ignore_index=True)
    previous_games.sort_values('date', inplace=True)
    
    previous_season_games = previous_games.loc[previous_games['season'] == season]
    home_previous_season_games = home_previous_games.loc[home_previous_games['season'] == season]
    away_previous_season_games = away_previous_games.loc[away_previous_games['season'] == season]
    
    return previous_season_games, home_previous_season_games, away_previous_season_games

def get_team_previous_games_stats(team, season, game_date, scenario, n_last_games, fixtures_df):
    response = get_team_previous_games(team, game_date, season, fixtures_df)
    if not response: return None
    
    previous_season_games, home_previous_season_games, away_previous_season_games = response
    
    total_games = len(previous_season_games.index)
    if total_games < 10 or (len(home_previous_season_games.index) < 5 and scenario == 'H') or (len(away_previous_season_games.index) < 5 and scenario == 'A'):
        return
    
    points_pct, win_pct, draw_pct, loss_pct, home_wins, home_draws, home_losses, away_wins, away_draws, away_losses = get_historical_stats(previous_season_games, home_previous_season_games, away_previous_season_games)
    
    previous_last_games = previous_season_games.iloc[-n_last_games:,:]
    home_last_games = previous_last_games.loc[previous_last_games['scenario'] == 'H']
    away_last_games = previous_last_games.loc[previous_last_games['scenario'] == 'A']
    
    points_pct_last_games, win_pct_last_games, draw_pct_last_games, loss_pct_last_games, home_wins_last_games, home_draws_last_games, home_losses_last_games, away_wins_last_games, away_draws_last_games, away_losses_last_games = get_historical_stats(previous_last_games, home_last_games, away_last_games)
    
    if scenario == 'H':
        ha_win_pct = home_wins * 100 / len(home_previous_season_games.index)
        ha_draw_pct = home_draws * 100 / len(home_previous_season_games.index)
        ha_loss_pct = home_losses * 100 / len(home_previous_season_games.index)
    else:
        ha_win_pct = away_wins * 100 / len(away_previous_season_games.index)
        ha_draw_pct = away_draws * 100 / len(away_previous_season_games.index)
        ha_loss_pct = away_losses * 100 / len(away_previous_season_games.index)
        
    game_stats = get_stats_mean(previous_season_games, n_last_games, scenario)
    
    if any([np.isnan(s) for s in game_stats]): return None
    
    return [points_pct, win_pct, draw_pct, loss_pct, ha_win_pct, ha_draw_pct, ha_loss_pct, win_pct_last_games, draw_pct_last_games, loss_pct_last_games] + game_stats

def build_formatted_csv(data_model):
    columns = ['game_date', 'season', 'home_team', 'away_team', 'home_odds', 'away_odds', 'draw_odds', 
           'home_pts_pct', 'home_win_pct', 'home_draw_pct', 'home_loss_pct', 'home_home_win_pct', 'home_home_draw_pct', 'home_home_loss_pct', 'home_win_pct_last_games', 'home_draw_pct_last_games', 'home_loss_pct_last_games', 'home_score_last_games', 'home_conceded_last_games', 'home_xg_last_games', 'home_conceded_xg_last_games', 
           'away_pts_pct', 'away_win_pct', 'away_draw_pct', 'away_loss_pct', 'away_away_win_pct', 'away_away_draw_pct', 'away_away_loss_pct', 'away_win_pct_last_games', 'away_draw_pct_last_games', 'away_loss_pct_last_games', 'away_score_last_games', 'away_conceded_last_games', 'away_xg_last_games', 'away_conceded_xg_last_games', 
           'outcome', 'home_score', 'away_score']
    data_df = pd.DataFrame(data_model, columns=columns)
    return data_df