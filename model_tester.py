import utils.predictor_functions as pf

league = 'mls'
seasons = '2018-2024'
season_start = 2019
season_end = 2023

for season in range(season_start, season_end):
    # Read the data
    X_full, y, X_test_full, y_test, odds_test = pf.get_league_data(league, seasons, season)

    # Define categorical and numerical columns
    categorical_cols, numerical_cols = pf.set_numerical_categorical_cols(X_full)

    # Keep selected columns only
    X_train, y_train, X_test = pf.filter_datasets(X_full, y, X_test_full, categorical_cols, numerical_cols)

    # Transform numerical and categorical cols
    X_train = pf.transform_x(X_train, categorical_cols, numerical_cols)
    X_test = pf.transform_x(X_test, categorical_cols, numerical_cols)

    # Get mi scores
    first_mi_scores = pf.make_mi_scores(X_train, y_train)

    # Create cluster features
    pf.create_cluster_features(X_train, X_test, first_mi_scores)

    # Get mi scores including the cluster features
    second_mi_scores = pf.make_mi_scores(X_train, y_train)

    # Create PCA features
    X_train, X_test = pf.apply_pca_datasets(X_train, X_test, second_mi_scores)

    # Run the random search algorithm for a few machine learning models
    pf.run_random_search(X_train, y_train, season, league)