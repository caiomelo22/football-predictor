import requests
from datetime import datetime as dt, timedelta
from bs4 import BeautifulSoup as soup

class BetExplorerService():
    def __init__(self, country, league):
        self.country = country
        self.league = league

    def replace_unique_team_cases(self, team):
        if team == "New York City":
            team = "NYFC"
        if team == "Atletico-MG":
            team = "Atlético Mineiro"
        if team == "Athletico-PR":
            team = "Ath Paranaense"
        if team == "Atletico GO":
            team = "Atl Goianiense"

        return team

    def get_next_games(self):
        url = f"https://www.betexplorer.com/football/{self.country}/{self.league}/"

        result = requests.get(url)

        doc = soup(result.text, "html.parser")

        table = doc.find("table", {"class": "table-main"})
        games = table.find_all("tr")[1:]

        games_lst = []

        for game in games:
            try:
                game_dict = dict()

                tds = game.find_all("td")

                if len(tds) < 9:
                    continue

                _, matchup_td, _, _, _, home_odds_td, draw_odds_td, away_odds_td, date_td = tds

                if not home_odds_td.find("button"):
                    continue

                teams_str_splitted = matchup_td.text.split(' - ')

                game_dict["home_team"] = self.replace_unique_team_cases(teams_str_splitted[0])
                game_dict["away_team"] = self.replace_unique_team_cases(teams_str_splitted[1])

                game_dict["home_odds"] = float(home_odds_td.find("button")["data-odd"])
                game_dict["draw_odds"] = float(draw_odds_td.find("button")["data-odd"])
                game_dict["away_odds"] = float(away_odds_td.find("button")["data-odd"])

                date_str = date_td.text.split(' ')[0]
                if date_str == 'Today':
                    game_dict["date"] = dt.now()
                elif date_str == 'Tomorrow':
                    game_dict["date"] = dt.now() + timedelta(days=1)
                else:
                    game_dict["date"] = dt.strptime(date_str + str(dt.now().year), '%d.%m.%Y')

                games_lst.append(game_dict)
            except Exception as e:
                print("Exception when fetching a game:", str(e))
                continue

        return games_lst