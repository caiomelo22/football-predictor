
import pandas as pd
import time
from thefuzz import fuzz
from datetime import timedelta, datetime as dt
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from .filtered_columns import selected_stats

def initialize_driver():
    option = Options()
    # option.headless = True
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    # driver.maximize_window()
    return driver

def get_teams_squad_id(home_td_index, tds):
    if len(tds[home_td_index].find_element(By.TAG_NAME, 'a').get_attribute("href").split('/')) > 7:
        squad_id_index = -3
    else:
        squad_id_index = -2
    home_squad_id = tds[home_td_index].find_element(By.TAG_NAME, 'a').get_attribute("href").split('/')[squad_id_index]
    away_squad_id = tds[home_td_index+4].find_element(By.TAG_NAME, 'a').get_attribute("href").split('/')[squad_id_index]
    return home_squad_id, away_squad_id

def scrape_fbref(league, league_id):
    driver = initialize_driver()

    data_model = []
    
    url = f"https://fbref.com/en/comps/{league_id}/schedule/{league}-Scores-and-Fixtures"
    driver.get(url)

    driver.maximize_window()

    fb = driver.find_element(By.CLASS_NAME, 'fb')
    rows = fb.find_elements(By.XPATH, '//table/tbody/tr')

    seasons_squad_ids = []
    total_games = 0
    for i, r in enumerate(rows):
        print(f"{i}/{len(rows)}")
        if not r.text: continue
        tds = r.find_elements(By.XPATH, './/child::td')
        week = r.find_element(By.XPATH, './/child::th').text
            
        if len(tds) == 12:
            date, _, home_team, home_xg, score, away_xg, away_team, _, _, _, _, _ = [t.text for t in tds]
            home_squad_id, away_squad_id = get_teams_squad_id(2, tds)
        elif len(tds) == 13:
            _, date, _, home_team, home_xg, score, away_xg, away_team, _, _, _, _, _ = [t.text for t in tds]
            home_squad_id, away_squad_id = get_teams_squad_id(3, tds)
        elif len(tds) == 14:
            _, _, date, _, home_team, home_xg, score, away_xg, away_team, _, _, _, _, _ = [t.text for t in tds]
            home_squad_id, away_squad_id = get_teams_squad_id(4, tds)
        else: continue

        if not home_xg and not away_xg:
            home_xg, away_xg = None, None
            
        if not score:
            home_score, away_score = None, None
        else:
            home_score, away_score = score.split('–')

        today = dt.now()
        tomorrow_date = (today + timedelta(days=1)).date()
        date_converted = dt.strptime(date, "%Y-%m-%d").date()
        if date_converted > tomorrow_date: break
        # elif date_converted >= today.date() and date_converted <= tomorrow_date:
        
        seasons_squad_ids.extend([(home_squad_id, home_team), (away_squad_id, away_team)])
        match_info = [date, week, home_team, home_xg, home_score, away_score, away_xg, away_team]
        data_model.append(match_info)
        total_games += 1

    driver.close()

    print(f"Total games:", len(data_model))
    
    return data_model, set(seasons_squad_ids)

def transform_odds_date(date):
    return dt.strptime(date, '%d.%m.%Y')

def set_fuzz_score(home_team, away_team, row):
    home_score = fuzz.ratio(row["home_team"], home_team)
    away_score = fuzz.ratio(row["away_team"], away_team)
    return home_score + away_score

def scrape_betexplorer(season_games, league, league_country):
    driver = initialize_driver()

    data_model = []
    
    url = f"https://www.betexplorer.com/football/{league_country}/{league}/"
    driver.get(url)

    driver.maximize_window()

    table = driver.find_element(By.XPATH, '/html/body/div[4]/div[5]/div/div/div[1]/section/div[3]/div/table')
    rows = table.find_elements(By.XPATH, './/tbody/tr')

    total_games = 0
    for i, r in enumerate(rows):
        print(f"{i}/{len(rows)}")
        if not r.text: continue
        tds = r.find_elements(By.XPATH, './/child::td')
        if len(tds) < 6: continue
        _, matchup, _, _, _, home_odds, draw_odds, away_odds, date = [t.text for t in tds]
        
        try:
            if not matchup: continue
            home_team, away_team = matchup.split(' - ')
        
            date = date.split(' ')[0]
            if date == 'Today':
                date = dt.now()
            elif date == 'Tomorrow':
                date = dt.now() + timedelta(days=1)
            elif not date.split('.')[-1]:
                date += str(dt.now().year)
                date = transform_odds_date(date)
                
            match_info = [date, home_team, home_odds, away_team, away_odds, draw_odds]
            data_model.append(match_info)
            total_games += 1
        except Exception as e:
            continue

    driver.close()

    columns = ['date', 'home_team', 'home_odds', 'away_team', 'away_odds', 'draw_odds']
    odds_df = pd.DataFrame(data_model, columns=columns)

    season_games["date"] = pd.to_datetime(season_games["date"])
    
    season_games["home_odds"] = None
    season_games["away_odds"] = None
    season_games["draw_odds"] = None
    
    for i, row in season_games.iterrows():
        print(f"{i}/{len(season_games)}")
        
        try:
            plus_one_day = row['date'] + timedelta(days=1)
            minus_one_day = row['date'] - timedelta(days=1)
            same_date_matches = odds_df[(odds_df['date'].dt.date == row['date'].date()) | (minus_one_day.date() == odds_df['date'].dt.date) | (plus_one_day.date() == odds_df['date'].dt.date)].reset_index(drop=True)
            if not len(same_date_matches): continue
            same_date_matches['matchup_score'] = same_date_matches.apply(lambda x: set_fuzz_score(row['home_team'], row['away_team'], x), axis=1)
            same_date_matches = same_date_matches.sort_values(by='matchup_score', ascending=False).reset_index(drop=True)
            match = same_date_matches.iloc[0]
            
            season_games.at[i, "home_odds"] = match['home_odds']
            season_games.at[i, "away_odds"] = match['away_odds']
            season_games.at[i, "draw_odds"] = match['draw_odds']
            
        except:
            continue

def get_value(attr, tds, cols):
    col_index = cols.index(attr)
    return tds[col_index-1].text

def save_game_stats(team, opp_team, date, venue, stats, cols, games_dict):
    if venue == 'Home':
        home_team, away_team = team, opp_team
        prefixed_cols = ['home_' + col for col in cols]
    else:
        away_team, home_team = team, opp_team
        prefixed_cols = ['away_' + col for col in cols]
    
    stats_dict = {col: stat for col, stat in zip(prefixed_cols, stats)}
    game_key = (home_team, away_team, date)
    if game_key in games_dict:
        games_dict[game_key].update(stats_dict)
    else:
        games_dict[game_key] = stats_dict
    
def scrape_advanced_stats(league_id, season_test, seasons_squad_ids):
    driver = initialize_driver()
    games_stats_dict = {}
    for squad_idx, si in enumerate(seasons_squad_ids):
        squad_id, team_name = si
        for stat_type in selected_stats.keys():
            print(f"{squad_idx}/{len(seasons_squad_ids)-1} --> {team_name}:{stat_type}")
            url = f"https://fbref.com/en/squads/{squad_id}/{season_test}/matchlogs/c{league_id}/{stat_type}"
            print(url)
            driver.get(url)

            rows = driver.find_elements(By.XPATH, '//table/tbody/tr')
            thead = driver.find_elements(By.XPATH, '//table/thead/tr')[1]
            cols = thead.find_elements(By.XPATH, './/child::th')
            cols = [c.text for c in cols]

            for i, r in enumerate(rows):
                if not r.text: continue
                tds = r.find_elements(By.XPATH, './/child::td')
                if not len(tds): continue
                date = r.find_element(By.XPATH, './/child::th').text

                opp_team = get_value('Opponent', tds, cols)
                venue = get_value('Venue', tds, cols)

                stats = []
                for stat_col in selected_stats[stat_type]:
                    stats.append(get_value(stat_col, tds, cols))

                save_game_stats(team_name, opp_team, date, venue, stats, selected_stats[stat_type], games_stats_dict)

            time.sleep(6)

    driver.close()

    return games_stats_dict

def complete_stats(game_stats, reg_cols, games_stats_dict):
    reg_dict = {col: stat for col, stat in zip(reg_cols, game_stats)}
    game_key = (reg_dict['home_team'], reg_dict['away_team'], reg_dict['date'])
    advanced_stats_dict = games_stats_dict.get(game_key)
    if not advanced_stats_dict: return reg_dict

    game_dict = {**reg_dict, **advanced_stats_dict}
    return game_dict