import pandas as pd
from config import api_football_key, conn_host, conn_database, conn_user, conn_password
import mysql.connector

def get_team_previous_games(team_id, game_date, season, fixtures_df):
    home_previous_games = fixtures_df.loc[(fixtures_df['home_id'] == team_id) & (fixtures_df['date'] < game_date)]
    away_previous_games = fixtures_df.loc[(fixtures_df['away_id'] == team_id) & (fixtures_df['date'] < game_date)]
    
    if len(home_previous_games.index) == 0 or len(away_previous_games.index) == 0:
        return None
    
    home_previous_games.rename(columns = {'home_id': 'team_id', 'home_team': 'team_name',
       'home_score': 'team_score', 'home_shots_on_goal': 'team_shots_on_goal', 'home_shots_off_goal': 'team_shots_off_goal', 'home_total_shots': 'team_total_shots', 'home_blocked_shots': 'team_blocked_shots',
       'home_shots_inside_box': 'team_shots_inside_box', 'home_shots_outside_box': 'team_shots_outside_box', 'home_fouls': 'team_fouls', 'home_corners': 'team_corners', 'home_offsides': 'team_offsides',
       'home_possession': 'team_possession', 'home_yellow_cards': 'team_yellow_cards', 'home_red_cards': 'team_red_cards', 'home_saves': 'team_saves', 'home_total_passes': 'team_total_passes',
       'home_passes_accurate': 'team_passes_accurate', 'home_passes_pct': 'team_passes_pct',
                                          
       'away_id': 'opp_id', 'away_team': 'opp_name', 
       'away_score': 'opp_score', 'away_shots_on_goal': 'opp_shots_on_goal', 'away_shots_off_goal': 'opp_shots_off_goal', 'away_total_shots': 'opp_total_shots', 'away_blocked_shots': 'opp_blocked_shots',
       'away_shots_inside_box': 'opp_shots_inside_box', 'away_shots_outside_box': 'opp_shots_outside_box', 'away_fouls': 'opp_fouls', 'away_corners': 'opp_corners', 'away_offsides': 'opp_offsides',
       'away_possession': 'opp_possession', 'away_yellow_cards': 'opp_yellow_cards', 'away_red_cards': 'opp_red_cards', 'away_saves': 'opp_saves', 'away_total_passes': 'opp_total_passes',
       'away_passes_accurate': 'opp_passes_accurate', 'away_passes_pct': 'opp_passes_pct',
                                          
       'home_odds': 'team_odds', 'away_odds': 'opp_odds'}, inplace=True)
    home_previous_games['scenario'] = 'H'
    
    away_previous_games.rename(columns = {'away_id': 'team_id', 'away_team': 'team_name',
       'away_score': 'team_score', 'away_shots_on_goal': 'team_shots_on_goal', 'away_shots_off_goal': 'team_shots_off_goal', 'away_total_shots': 'team_total_shots', 'away_blocked_shots': 'team_blocked_shots',
       'away_shots_inside_box': 'team_shots_inside_box', 'away_shots_outside_box': 'team_shots_outside_box', 'away_fouls': 'team_fouls', 'away_corners': 'team_corners', 'away_offsides': 'team_offsides',
       'away_possession': 'team_possession', 'away_yellow_cards': 'team_yellow_cards', 'away_red_cards': 'team_red_cards', 'away_saves': 'team_saves', 'away_total_passes': 'team_total_passes',
       'away_passes_accurate': 'team_passes_accurate', 'away_passes_pct': 'team_passes_pct',
                                          
       'home_id': 'opp_id', 'home_team': 'opp_name', 
       'home_score': 'opp_score', 'home_shots_on_goal': 'opp_shots_on_goal', 'home_shots_off_goal': 'opp_shots_off_goal', 'home_total_shots': 'opp_total_shots', 'home_blocked_shots': 'opp_blocked_shots',
       'home_shots_inside_box': 'opp_shots_inside_box', 'home_shots_outside_box': 'opp_shots_outside_box', 'home_fouls': 'opp_fouls', 'home_corners': 'opp_corners', 'home_offsides': 'opp_offsides',
       'home_possession': 'opp_possession', 'home_yellow_cards': 'opp_yellow_cards', 'home_red_cards': 'opp_red_cards', 'home_saves': 'opp_saves', 'home_total_passes': 'opp_total_passes',
       'home_passes_accurate': 'opp_passes_accurate', 'home_passes_pct': 'opp_passes_pct',
                                          
       'home_odds': 'opp_odds', 'away_odds': 'team_odds'}, inplace=True)
    away_previous_games['scenario'] = 'A'
    
    previous_games = pd.concat([home_previous_games, away_previous_games], axis=0, ignore_index=True)
    previous_games.sort_values('date', inplace=True)
    
    previous_season_games = previous_games.loc[previous_games['season'] == season]
    home_previous_season_games = home_previous_games.loc[home_previous_games['season'] == season]
    away_previous_season_games = away_previous_games.loc[away_previous_games['season'] == season]
    
    return previous_season_games, home_previous_season_games, away_previous_season_games

def get_games_results(games, scenario):
    loser = 'A' if scenario == 'H' else 'H'
    return len(games.loc[games['winner'] == scenario].index), len(games.loc[games['winner'] == 'D'].index), len(games.loc[games['winner'] == loser].index)

def get_stats_mean(games, team_id, n_last_games, scenario):
    games = games.iloc[-n_last_games:,:]
    
    team_stats = [games['team_score'].mean(), games['opp_score'].mean(), games['team_shots_on_goal'].mean(), games['team_shots_off_goal'].mean(),
                 games['team_total_shots'].mean(), games['team_blocked_shots'].mean(), games['team_shots_inside_box'].mean(),
                 games['team_shots_outside_box'].mean(), games['team_fouls'].mean(), games['team_corners'].mean(),
                 games['team_offsides'].mean(), games['team_possession'].mean(), games['team_yellow_cards'].mean(),
                 games['team_red_cards'].mean(), games['team_saves'].mean(), games['team_total_passes'].mean(),
                 games['team_passes_accurate'].mean(), games['team_passes_pct'].mean()]
#     opp_stats = [games['opp_shots_on_goal'].mean(), games['opp_shots_off_goal'].mean(),
#                  games['opp_total_shots'].mean(), games['opp_blocked_shots'].mean(), games['opp_shots_inside_box'].mean(),
#                  games['opp_shots_outside_box'].mean(), games['opp_fouls'].mean(), games['opp_corners'].mean(),
#                  games['opp_offsides'].mean(), games['opp_possession'].mean(), games['opp_yellow_cards'].mean(),
#                  games['opp_red_cards'].mean(), games['opp_saves'].mean(), games['opp_total_passes'].mean(),
#                  games['opp_passes_accurate'].mean(), games['opp_passes_pct'].mean()]
    
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
    

def get_team_previous_games_stats(team_id, season, game_date, scenario, n_last_games, fixtures_df):
    response = get_team_previous_games(team_id, game_date, season, fixtures_df)
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
        
    game_stats = get_stats_mean(previous_season_games, team_id, n_last_games, scenario)
    
    return [points_pct, win_pct, draw_pct, loss_pct, ha_win_pct, ha_draw_pct, ha_loss_pct, win_pct_last_games, draw_pct_last_games, loss_pct_last_games] + game_stats

def connect_to_db():
    return mysql.connector.connect(host=conn_host, 
                                     database=conn_database,
                                     user=conn_user,
                                     password=conn_password)

def execute_query(query, read_only = True):
    resp = None
    try:
        db = connect_to_db()
        if read_only:
            resp = pd.read_sql_query(query, db)
        else:
            mycursor = db.cursor()
            mycursor.execute(query)

            db.commit()
    except Exception as e:
        print(e)
    db.close()
    return resp

def execute_multiple_queries(queries):
    try:
        db = connect_to_db()
        mycursor = db.cursor()
        for query in queries:
            mycursor.execute(query)

        db.commit()
        db.close()
    except Exception as e:
        print(e)