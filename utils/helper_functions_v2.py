import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_league_data(league, seasons, season_test):
    # Read the data
    X_full = pd.read_csv(f'./leagues_data/{league}/{seasons}.csv', index_col=0)
    X_test_full = X_full[X_full['season'] == season_test]
    X_full = X_full[X_full['season'] < season_test]

    # Remove rows with missing target, separate target from predictors
    y = X_full.outcome
    X_full.drop(['outcome', 'home_score', 'away_score', 'home_odds',
                'away_odds', 'draw_odds'], axis=1, inplace=True)

    y_test = X_test_full.outcome
    odds_test = X_test_full[['home_odds', 'away_odds', 'draw_odds']]
    X_test_full.drop(['outcome', 'home_score', 'away_score',
                     'home_odds', 'away_odds', 'draw_odds'], axis=1, inplace=True)

    return X_full, y, X_test_full, y_test, odds_test


def set_numerical_categorical_cols(X):
    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X.columns if
                        X[cname].nunique() < 10 and
                        X[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X.columns if
                      X[cname].dtype in ['int64', 'float64']]

    return categorical_cols, numerical_cols


def filter_datasets(X_full, y, X_test_full, categorical_cols, numerical_cols):
    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    y_train = y.copy()
    X_train = X_full[my_cols]
    X_test = X_test_full[my_cols]

    return X_train, y_train, X_test


def transform_x(X, categorical_cols, numerical_cols):
    X = X.copy()

    # Replacing missing values
    imputer = SimpleImputer(strategy='mean')
    X[numerical_cols] = imputer.fit_transform(X[numerical_cols])

    if categorical_cols:
        imputer = SimpleImputer(strategy='constant')
        X[categorical_cols] = imputer.fit_transform(X[categorical_cols])

    # Label encoding for categoricals
    for colname in categorical_cols:
        X[colname], _ = X[colname].factorize()

    return X


def make_mi_scores(X, y):
    # All discrete features should now have integer dtypes
    discrete_features = X.dtypes == int

    mi_scores = mutual_info_classif(
        X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    plt.figure(figsize=(12, 12))
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


def plot_feature_corr_chart(X, numerical_cols):
    plt.figure(figsize=(12, 12))
    plot_data = X[numerical_cols].corr()
    sns.heatmap(data=plot_data)
    plt.show()


def scale_values(X, features_to_explore):
    # Standardize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.loc[:, features_to_explore])
    return X_scaled


def create_cluster_features(X_train, X_test, mi_scores):  # First mi scores
    features_to_explore = [mi_scores.index[f] for f in range(len(mi_scores))]
    print('Total features to consider when clustering:', len(features_to_explore))

    for i in range(len(features_to_explore)):
        X_scaled = scale_values(X_train, features_to_explore[i:i+2])
        X_test_scaled = scale_values(X_test, features_to_explore[i:i+2])

        kmeans = KMeans(n_clusters=5, n_init=10, random_state=0)
        X_train[f"Cluster_{i+1}"] = kmeans.fit_predict(X_scaled)
        X_test[f"Cluster_{i+1}"] = kmeans.predict(X_test_scaled)

        X_train[f"Cluster_{i+1}"] = X_train[f"Cluster_{i+1}"].astype('int')
        X_test[f"Cluster_{i+1}"] = X_test[f"Cluster_{i+1}"].astype('int')


def get_pca(X, cols, pca, just_transform=False):
    # Standardize
    X_scaled = scale_values(X, cols)

    if just_transform:
        X_pca = pca.transform(X_scaled)
    else:
        X_pca = pca.fit_transform(X_scaled)

    # Convert to df
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names, index=X.index)

    X_final = pd.concat([X, X_pca], axis=1)
    return X_final


def apply_pca_datasets(X_train, X_test, mi_scores, min_mi_score=0.001):  # Second mi scores
    features_to_explore = [mi_scores.index[f] for f in range(
        len(mi_scores)) if mi_scores[f] > min_mi_score]
    print('Total features to consider when doing the PCA:',
          len(features_to_explore))

    pca = PCA(random_state=0)
    X_train = get_pca(X_train, features_to_explore, pca)
    X_test = get_pca(X_test, features_to_explore, pca, just_transform=True)

    return X_train, X_test


def run_grid_search(X, y, model, parameters_grid, cv=5, verbose=0):
    print('Running the grid search algorithm to get the best possible model...')
    grid_search_cv = GridSearchCV(estimator=model, param_grid=parameters_grid,
                                          cv=cv, verbose=verbose)

    grid_search_cv.fit(X, y)

    best_random = grid_search_cv.best_estimator_
    best_parameters = grid_search_cv.cv_results_

    print('Best random:', best_random)

    return best_random, best_parameters


def run_random_forest_grid_search(X, y):
    # Random Forest Optimizer
    model = RandomForestClassifier(random_state=0)
    # Number of features to consider at every split
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['sqrt', 'log2', None]
    # Criterion
    criterion = ['gini', 'entropy', 'log_loss']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    parameters_grid = {'n_estimators': n_estimators,
                       'criterion': criterion,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

    return run_grid_search(X, y, model, parameters_grid)


def run_gradient_boosting_grid_search(X, y):
    # Gradient Boosting Optimizer
    model = GradientBoostingClassifier(random_state=0)
    # Criterion
    criterion = ['friedman_mse', 'squared_error']
    # Losse functions
    loss = ['log_loss']
    # Learning rates
    learning_rate = [x for x in np.linspace(0.05, 0.3, 6)]
    # Number of estimators
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Max features
    max_features = ['sqrt', 'log2', None]

    parameters_grid = {'n_estimators': n_estimators,
                       'criterion': criterion,
                       'max_features': max_features,
                       'loss': loss,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'learning_rate': learning_rate}

    return run_grid_search(X, y, model, parameters_grid)

def build_pipeline(X_train, y_train, model):
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='median')

    categorical_cols, numerical_cols = set_numerical_categorical_cols(X_train)

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Define normalization function
    scaler = MinMaxScaler()

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols),
            ('normalization', scaler, numerical_cols + categorical_cols)
        ])

    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)
                                ])

    # Preprocessing of training data, fit model 
    my_pipeline.fit(X_train, y_train)
    
    return my_pipeline

def get_match_profit(row):
    if row['outcome'] == row['pred']:
        if row['pred'] == 'H': return row['home_odds'] - 1
        elif row['pred'] == 'A': return row['away_odds'] - 1
        elif row['pred'] == 'D': return row['draw_odds'] - 1
    else:
        return -1

def build_pred_df(my_pipeline, X_test, y_test, odds_test):
    preds_test = my_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, preds_test)
    print('Accuracy:', accuracy)

    test_probs = my_pipeline.predict_proba(X_test)
    probs_test_df = pd.DataFrame(test_probs, index=y_test.index, columns=['away_probs', 'draw_probs', 'home_probs'])
    preds_test_df = pd.DataFrame(preds_test, index=y_test.index, columns=['pred'])
    test_results_df = pd.concat([y_test, preds_test_df, probs_test_df, odds_test], axis=1)

    test_results_df['profit'] = test_results_df.apply(lambda x: get_match_profit(x), axis=1)
    print('Model profit:', test_results_df.profit.sum())
    negative_consecutive_count = test_results_df['profit'].lt(0).astype(int).groupby((test_results_df['profit'] >= 0).cumsum()).sum().max()
    print('Maximum negative sequence: ', negative_consecutive_count)
    positive_consecutive_count = test_results_df['profit'].gt(0).astype(int).groupby((test_results_df['profit'] < 0).cumsum()).sum().max()
    print('Maximum positive sequence: ', positive_consecutive_count)

    return test_results_df

