import utils.predictor_functions as pf
import json
import pandas as pd
from joblib import load
import pickle
import warnings
import joblib
import os
warnings.filterwarnings('ignore')
print("Setup Complete")

save_pipeline = True
run_random_search = False
league = 'mls'
seasons = '2018-2024'
season_test = 2023

# Read the data
X_full, y, X_test_full, y_test, odds_test = pf.get_league_data(league, seasons, season_test)

# Define categorical and numerical columns
categorical_cols, numerical_cols = pf.set_numerical_categorical_cols(X_full)

# Keep selected columns only
X_train, y_train, X_test = pf.filter_datasets(X_full, y, X_test_full, categorical_cols, numerical_cols)

# First mi scores
first_mi_scores = pf.make_mi_scores(X_train, y_train)

# Cluster features
X_train, X_test, kmeans_scaler_list, features_kmeans_list = pf.create_cluster_features(X_train, X_test, first_mi_scores)

# Second mi scores
second_mi_scores = pf.make_mi_scores(X_train, y_train)

# PCA features
X_train, X_test, pca_features, pca_scaler, pca = pf.apply_pca_datasets(X_train, X_test, second_mi_scores)

def won_bet(row):
    return 1 if row['profit'] > 0 else 0

dir_path = f"leagues_v2/{league}/best_models/{season_test}"
models_path = f"{dir_path}/best_models.json"
with open(models_path, 'rb') as file:
    models_dict = json.load(file)
    
models = models_dict.keys()
voting_classifier_estimators = []
for model in models:
    model_path = f"{dir_path}/{model}.joblib"
    try:
        loaded_model = load(model_path)
    except:
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
    models_dict[model]['estimator'] = loaded_model

# Only predicting after 15 team games. It's shown that is more profitable
X_test_filtered = X_test.reset_index(drop=True)[50:]
y_test_filtered = y_test.reset_index(drop=True)[50:]
odds_test_filtered = odds_test.reset_index(drop=True)[50:]

progress_data = []
for model in models_dict.keys():
    print(f"Results for model {model}:")
    score = models_dict[model].get('score')
    if score:
        print('Training score:', score)
    my_pipeline = pf.build_pipeline(X_train, y_train, models_dict[model]['estimator'])
    if not len(X_test_filtered): continue
    test_results_df = pf.build_pred_df(my_pipeline, X_test_filtered, y_test_filtered, odds_test_filtered)
    test_results_df['won'] = test_results_df.apply(lambda x: won_bet(x), axis=1)
    total_won = test_results_df['won'].sum()
    progress_data.append([test_results_df['profit'].sum(), score, total_won/len(test_results_df)])
    
cols = ['profit', 'training_score', 'test_score']
profit_df = pd.DataFrame(progress_data, columns=cols, index=models_dict.keys())

print(profit_df)

# Since the last pipeline was the Voting Classifier one, let's save it
# If you want another, change some of the code above
if save_pipeline:
    path = f"leagues_v2/{league}/official"
    if not os.path.exists(path):
        os.makedirs(path)
    joblib.dump(my_pipeline, f"{path}/pipeline.joblib")
    joblib.dump(kmeans_scaler_list, f"{path}/kmeans_scaler_list.joblib")
    joblib.dump(features_kmeans_list, f"{path}/features_kmeans_list.joblib")
    joblib.dump(pca_features, f"{path}/pca_features.joblib")
    joblib.dump(pca_scaler, f"{path}/pca_scaler.joblib")
    joblib.dump(pca, f"{path}/pca.joblib")