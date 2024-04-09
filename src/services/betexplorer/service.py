import requests
from bs4 import BeautifulSoup as soup

class BetExplorerService():
    def __init__(self, country, league):
        self.country = country
        self.league = league

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

                teams_str = tds[1].text
                teams_str_splitted = teams_str.split(' - ')

                game_dict["home_team"] = teams_str_splitted[0]
                game_dict["away_team"] = teams_str_splitted[1]

                game_dict["home_odds"] = float(tds[5].find("button")["data-odd"])
                game_dict["draw_odds"] = float(tds[6].find("button")["data-odd"])
                game_dict["away_odds"] = float(tds[7].find("button")["data-odd"])

                games_lst.append(game_dict)
            except Exception as e:
                print("Exception when fetching a game:", str(e))
                continue

        return games_lst