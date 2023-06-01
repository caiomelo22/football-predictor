
import pandas as pd
import io
import json
import os
from thefuzz import fuzz
from datetime import timedelta, datetime as dt
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def scrape_fbref(league, league_id):
    option = Options()
    # option.headless = True
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    # driver.maximize_window()

    data_model = []
    
    url = f"https://fbref.com/en/comps/{league_id}/schedule/{league}-Scores-and-Fixtures"
    driver.get(url)

    driver.maximize_window()

    fb = driver.find_element(By.CLASS_NAME, 'fb')
    rows = fb.find_elements(By.XPATH, '//table/tbody/tr')

    total_games = 0
    for i, r in enumerate(rows):
        print(f"{i}/{len(rows)}")
        if not r.text: continue
        tds = r.find_elements(By.XPATH, './/child::td')
        week = r.find_element(By.XPATH, './/child::th').text
            
        if len(tds) == 12:
            date, _, home_team, home_xg, score, away_xg, away_team, _, _, _, _, _ = [t.text for t in tds]
        elif len(tds) == 13:
            _, date, _, home_team, home_xg, score, away_xg, away_team, _, _, _, _, _ = [t.text for t in tds]
        elif len(tds) == 14:
            _, _, date, _, home_team, home_xg, score, away_xg, away_team, _, _, _, _, _ = [t.text for t in tds]
        else: continue

        if not home_xg and not away_xg:
            home_xg, away_xg = None, None
            
        if not score:
            home_score, away_score = None, None
        else:
            home_score, away_score = score.split('–')

        match_info = [date, week, home_team, home_xg, home_score, away_score, away_xg, away_team]
        match_str = f"{date} ({week}): {home_team} ({home_xg}) {home_score} x {away_score} ({away_xg}) {away_team}"
        data_model.append(match_info)
#         print(match_str)
        total_games += 1

    driver.close()

    print(f"Total games:", len(data_model))
    columns = ['date', 'week', 'home_team', 'home_xg', 'home_score', 'away_score', 'away_xg', 'away_team']
    season_df = pd.DataFrame(data_model, columns=columns)
    
    return season_df

def transform_odds_date(date):
    return dt.strptime(date, '%d.%m.%Y')

def set_fuzz_score(home_team, away_team, row):
    home_score = fuzz.ratio(row["home_team"], home_team)
    away_score = fuzz.ratio(row["away_team"], away_team)
    return home_score + away_score

def scrape_betexplorer(season_games, league, league_country):
    option = Options()
    # option.headless = True
    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    # driver.maximize_window()

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
                
            match_info = [transform_odds_date(date), home_team, home_odds, away_team, away_odds, draw_odds]
            data_model.append(match_info)
            total_games += 1
        except Exception as e:
            continue

    columns = ['date', 'home_team', 'home_odds', 'away_team', 'away_odds', 'draw_odds']
    odds_df = pd.DataFrame(data_model, columns=columns)

    season_games["date"] = pd.to_datetime(season_games["date"])
    
    season_games["home_odds"] = None
    season_games["away_odds"] = None
    season_games["draw_odds"] = None
    
    for i, row in season_games.iterrows():
        print(f"{i}/{len(season_games)}")
        
        try:
            plus_one_day = odds_df['date'] + timedelta(days=1)
            minus_one_day = odds_df['date'] - timedelta(days=1)
            same_date_matches = odds_df[(odds_df['date'] == row['date']) | (minus_one_day == row['date']) | (plus_one_day == row['date'])].reset_index(drop=True)
            same_date_matches['matchup_score'] = same_date_matches.apply(lambda x: set_fuzz_score(row['home_team'], row['away_team'], x), axis=1)
            same_date_matches = same_date_matches.sort_values(by='matchup_score', ascending=False).reset_index(drop=True)
            match = same_date_matches.iloc[0]

            season_games.at[i, "home_odds"] = match['home_odds']
            season_games.at[i, "away_odds"] = match['away_odds']
            season_games.at[i, "draw_odds"] = match['draw_odds']
            
        except:
            continue