import utils.helper_functions_v2 as hf

league = 'premier-league'
seasons = '2017-2023'
season_test = 2022

# Read the data
X_full, y, X_test_full, y_test, odds_test = hf.get_league_data(league, seasons, season_test)

# Define categorical and numerical columns
categorical_cols, numerical_cols = hf.set_numerical_categorical_cols(X_full)

# Keep selected columns only
X_train, y_train, X_test = hf.filter_datasets(X_full, y, X_test_full, categorical_cols, numerical_cols)

# Transform numerical and categorical cols
X_train = hf.transform_x(X_train, categorical_cols, numerical_cols)
X_test = hf.transform_x(X_test, categorical_cols, numerical_cols)

# Get mi scores
first_mi_scores = hf.make_mi_scores(X_train, y_train)

# Create cluster features
hf.create_cluster_features(X_train, X_test, first_mi_scores)

# Get mi scores including the cluster features
second_mi_scores = hf.make_mi_scores(X_train, y_train)

# Create PCA features
X_train, X_test = hf.apply_pca_datasets(X_train, X_test, second_mi_scores)

# Get mi scores including the PCA features
third_mi_scores = hf.make_mi_scores(X_train, y_train)

# Run the grid search algorithm for a few machine learning models
hf.run_random_search(X_train, y_train)