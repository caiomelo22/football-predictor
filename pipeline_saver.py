import utils.predictor_functions as pf
from utils.league_options import filtered_cols, selected_stats, strategy
import warnings
import joblib
import os
import json
warnings.filterwarnings('ignore')
print("Setup Complete")

save_pipeline = True
run_random_search = False
league = 'serie-a'
seasons = '2019-2024'
season_test = 2023
betting_starts_after_n_games = 0

# Read the data
X_full, y, X_test_full, y_test, odds_test = pf.get_league_data(league, seasons, season_test)
# Keep selected columns only
X_train, y_train, X_test = pf.filter_datasets(X_full, y, X_test_full)

# First mi scores
first_mi_scores = pf.make_mi_scores(X_train, y_train)

# Cluster features
X_train, X_test, kmeans_scaler_list, features_kmeans_list = pf.create_cluster_features(X_train, X_test, first_mi_scores)

# Second mi scores
second_mi_scores = pf.make_mi_scores(X_train, y_train)

# PCA features
X_train, X_test, pca_features, pca_scaler, pca = pf.apply_pca_datasets(X_train, X_test, second_mi_scores)

my_pipeline = pf.simulate(X_train, y_train, X_test, y_test, odds_test, betting_starts_after_n_games, verbose=1)

cols_info = {'filtered_cols': filtered_cols, 'selected_stats': selected_stats, 'strategy': strategy}

# Since the last pipeline was the Voting Classifier one, let's save it
# If you want another, change some of the code above
if save_pipeline:
    path = f"leagues/{league}/official"
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f"{path}/columns.json", 'w') as json_file:
        json.dump(cols_info, json_file)

    joblib.dump(my_pipeline, f"{path}/pipeline.joblib")
    joblib.dump(kmeans_scaler_list, f"{path}/kmeans_scaler_list.joblib")
    joblib.dump(features_kmeans_list, f"{path}/features_kmeans_list.joblib")
    joblib.dump(pca_features, f"{path}/pca_features.joblib")
    joblib.dump(pca_scaler, f"{path}/pca_scaler.joblib")
    joblib.dump(pca, f"{path}/pca.joblib")