import utils.regression.totals_predictor_functions as tpf
import warnings
warnings.filterwarnings('ignore')
print("Setup Complete")

save_pipeline = False
run_random_search = False
league = 'major-league-soccer'
seasons = '2018-2024'
season_test = 2023
betting_starts_after_n_games = 0

score_cols = ['home_score', 'away_score']

for col in score_cols:
    # Read the data
    X_train, y_train, X_test, y_test = tpf.get_league_data(league, seasons, season_test, col)

    tpf.regression_model_evaluation(X_train, y_train, X_test, y_test)